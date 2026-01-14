"""
Blosc2 codec for numcodecs.

This module provides a Blosc2 codec class that wraps python-blosc2,
supporting all blosc2 compressors including user-defined codecs like openh264.
"""

import numpy as np

from .abc import Codec
from .compat import ensure_contiguous_ndarray

# Shuffle constants (matching blosc2)
NOSHUFFLE = 0
SHUFFLE = 1
BITSHUFFLE = 2
AUTOSHUFFLE = -1

# Filter IDs
NOFILTER = 0
SHUFFLE_FILTER = 1
BITSHUFFLE_FILTER = 2

# OpenH264 codec ID (user-defined range)
OPENH264_CODEC_ID = 240


def _cube_side_from_len(length):
    """Calculate cube side from total element count. Returns -1 if not cubic."""
    if length <= 0:
        return -1
    root = round(length ** (1/3))
    if root * root * root == length:
        return root
    return -1


class Blosc2(Codec):
    """Codec providing compression using the Blosc2 meta-compressor.

    This codec wraps python-blosc2 and supports all blosc2 compressors,
    including user-defined codecs like 'openh264'.

    Parameters
    ----------
    cname : string, optional
        A string naming one of the compression algorithms available within blosc2.
        Standard compressors: 'blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd'.
        User-defined: 'openh264' (for H.264 video compression of 3D cubic chunks).
        Default is 'zstd'.
    clevel : integer, optional
        An integer between 0 and 9 specifying the compression level.
        Default is 5.
    shuffle : integer, optional
        Shuffle filter: NOSHUFFLE (0), SHUFFLE (1), BITSHUFFLE (2), or AUTOSHUFFLE (-1).
        Note: shuffle is ignored for openh264 codec.
        Default is SHUFFLE.
    blocksize : int, optional
        The requested size of the compressed blocks. If 0 (default), an automatic
        blocksize will be used. For openh264, this is automatically set to the
        chunk size.
    typesize : int, optional
        The size in bytes of uncompressed array elements. Default is 1 for openh264,
        auto-detected otherwise.
    qp : int, optional
        Quantization parameter for H.264 codec (0-51). Only used with cname='openh264'.
        0 = highest quality, 51 = highest compression.
        Default is 26.

    Notes
    -----
    When using cname='openh264':
    - Data must be uint8 (typesize=1)
    - Chunks must be cubic (NxNxN where N is even)
    - Compression is lossy (H.264 video codec)
    - blocksize is automatically set to match chunk size
    - qp controls quality/compression tradeoff (0-51)

    Examples
    --------
    Standard compression:
    >>> codec = Blosc2(cname='zstd', clevel=5)

    H.264 compression for 3D volumes:
    >>> codec = Blosc2(cname='openh264')

    H.264 with high compression (lower quality):
    >>> codec = Blosc2(cname='openh264', qp=39)

    """

    codec_id = 'blosc2'
    NOSHUFFLE = NOSHUFFLE
    SHUFFLE = SHUFFLE
    BITSHUFFLE = BITSHUFFLE
    AUTOSHUFFLE = AUTOSHUFFLE

    # Standard blosc2 compressor names
    STANDARD_COMPRESSORS = ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']

    # User-defined compressors with their codec IDs
    USER_COMPRESSORS = {
        'openh264': OPENH264_CODEC_ID,
    }

    def __init__(self, cname='zstd', clevel=5, shuffle=SHUFFLE, blocksize=0, typesize=None, qp=26):
        self.cname = cname
        self.clevel = clevel
        self.shuffle = shuffle
        self.blocksize = blocksize
        self.typesize = typesize
        self.qp = qp

        # Validate compressor name
        if cname not in self.STANDARD_COMPRESSORS and cname not in self.USER_COMPRESSORS:
            all_compressors = self.STANDARD_COMPRESSORS + list(self.USER_COMPRESSORS.keys())
            raise ValueError(
                f"Unknown compressor: {cname!r}. "
                f"Available: {all_compressors}"
            )

        # For openh264, enforce typesize=1
        if cname == 'openh264' and typesize is not None and typesize != 1:
            raise ValueError("openh264 codec requires typesize=1 (uint8 data)")

        # Validate QP range for openh264
        if cname == 'openh264':
            if not (0 <= qp <= 51):
                raise ValueError(f"QP must be 0-51 for openh264. Got {qp}")

    def _get_blosc2(self):
        """Lazy import of blosc2 module."""
        try:
            import blosc2
            return blosc2
        except ImportError:
            raise ImportError(
                "blosc2 is required for Blosc2 codec. "
                "Install with: pip install blosc2"
            )

    def encode(self, buf):
        """Compress data using blosc2.

        Parameters
        ----------
        buf : array-like
            Data to compress.

        Returns
        -------
        bytes
            Compressed data.
        """
        blosc2 = self._get_blosc2()

        # Ensure contiguous array
        arr = ensure_contiguous_ndarray(buf)
        data = arr.tobytes()

        # Determine typesize
        if self.typesize is not None:
            typesize = self.typesize
        elif hasattr(arr, 'itemsize'):
            typesize = arr.itemsize
        else:
            typesize = 1

        # Handle openh264 codec
        if self.cname == 'openh264':
            return self._encode_openh264(blosc2, data, arr.nbytes)

        # Standard blosc2 compression
        # Map shuffle constant
        if self.shuffle == AUTOSHUFFLE:
            if typesize == 1:
                filter_id = BITSHUFFLE_FILTER
            else:
                filter_id = SHUFFLE_FILTER
        elif self.shuffle == NOSHUFFLE:
            filter_id = NOFILTER
        elif self.shuffle == BITSHUFFLE:
            filter_id = BITSHUFFLE_FILTER
        else:
            filter_id = SHUFFLE_FILTER

        # Get codec ID for standard compressors
        codec_id = getattr(blosc2.Codec, self.cname.upper(), blosc2.Codec.ZSTD)

        compressed = blosc2.compress2(
            data,
            codec=codec_id,
            clevel=self.clevel,
            filter=filter_id,
            typesize=typesize,
            blocksize=self.blocksize if self.blocksize > 0 else 0,
        )

        return compressed

    def _encode_openh264(self, blosc2, data, nbytes):
        """Encode using openh264 codec.

        Parameters
        ----------
        blosc2 : module
            The blosc2 module.
        data : bytes
            Raw data to compress.
        nbytes : int
            Number of bytes.

        Returns
        -------
        bytes
            Compressed data in blosc2 chunk format.
        """
        # Validate cubic shape
        side = _cube_side_from_len(nbytes)
        if side < 0:
            raise ValueError(
                f"openh264 codec requires cubic chunks. "
                f"Got {nbytes} bytes which is not a perfect cube (NxNxN)."
            )

        if side % 2 != 0:
            raise ValueError(
                f"openh264 codec requires even chunk dimensions. "
                f"Got cube side={side}."
            )

        # For openh264, blocksize must equal chunk size
        # codec_meta is used to pass QP parameter to the openh264 codec
        compressed = blosc2.compress2(
            data,
            codec=OPENH264_CODEC_ID,
            codec_meta=self.qp,  # QP parameter for H.264
            typesize=1,
            blocksize=nbytes,  # Critical: must match chunk size
        )

        return compressed

    def decode(self, buf, out=None):
        """Decompress data using blosc2.

        Parameters
        ----------
        buf : bytes-like
            Compressed data.
        out : array-like, optional
            Output buffer.

        Returns
        -------
        bytes or ndarray
            Decompressed data.
        """
        blosc2 = self._get_blosc2()

        # Ensure bytes
        if not isinstance(buf, bytes):
            buf = bytes(buf)

        # Decompress
        decompressed = blosc2.decompress2(buf)

        if out is not None:
            out = ensure_contiguous_ndarray(out)
            out.flat[:] = np.frombuffer(decompressed, dtype=out.dtype)
            return out

        return decompressed

    def get_config(self):
        """Return codec configuration for serialization."""
        config = {
            'id': self.codec_id,
            'cname': self.cname,
            'clevel': self.clevel,
            'shuffle': self.shuffle,
            'blocksize': self.blocksize,
        }
        if self.typesize is not None:
            config['typesize'] = self.typesize
        if self.cname == 'openh264':
            config['qp'] = self.qp
        return config

    @classmethod
    def from_config(cls, config):
        """Create codec from configuration."""
        config = dict(config)
        config.pop('id', None)
        return cls(**config)

    def __repr__(self):
        parts = [
            f"cname={self.cname!r}",
        ]
        if self.cname == 'openh264':
            parts.append(f"qp={self.qp}")
        else:
            parts.append(f"clevel={self.clevel!r}")
            shuffle_names = {-1: 'AUTOSHUFFLE', 0: 'NOSHUFFLE', 1: 'SHUFFLE', 2: 'BITSHUFFLE'}
            parts.append(f"shuffle={shuffle_names.get(self.shuffle, self.shuffle)}")
        if self.blocksize != 0:
            parts.append(f"blocksize={self.blocksize}")
        if self.typesize is not None:
            parts.append(f"typesize={self.typesize}")
        return f"Blosc2({', '.join(parts)})"


def list_compressors():
    """List all available blosc2 compressors.

    Returns
    -------
    list
        List of compressor names.
    """
    return Blosc2.STANDARD_COMPRESSORS + list(Blosc2.USER_COMPRESSORS.keys())
