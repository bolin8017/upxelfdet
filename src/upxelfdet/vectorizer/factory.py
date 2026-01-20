"""Vectorizer factory module.

This module provides a factory for creating vectorizer instances based on
configuration.
"""

from ..constants import (
    VECTORIZE_METHOD_NGRAM_NUMERIC,
    VECTORIZE_METHOD_RAW_BYTES,
    VectorizationMethod,
)
from .base import BaseVectorizer
from .ngram_numeric import NGramNumericVectorizer
from .raw_bytes import RawBytesVectorizer


class VectorizerFactory:
    """Factory for creating vectorizer instances.

    Supported vectorization methods:
    - "ngram_numeric": N-gram numeric vectorizer with sparse matrix storage
    - "raw_bytes": Fixed-length raw byte sequence vectorizer
    """

    _VECTORIZERS: dict[str, type[BaseVectorizer]] = {
        VECTORIZE_METHOD_NGRAM_NUMERIC: NGramNumericVectorizer,
        VECTORIZE_METHOD_RAW_BYTES: RawBytesVectorizer,
    }

    @classmethod
    def create(cls, method: VectorizationMethod) -> BaseVectorizer:
        """Create a vectorizer instance.

        Args:
            method: Vectorization method name.

        Returns:
            A vectorizer instance.

        Raises:
            ValueError: If the method is not supported.
        """
        vectorizer_class = cls._VECTORIZERS.get(method)

        if vectorizer_class is None:
            supported = ", ".join(cls._VECTORIZERS.keys())
            raise ValueError(
                f"Unknown vectorization method: {method}. "
                f"Supported methods: {supported}"
            )

        return vectorizer_class()

    @classmethod
    def supported_methods(cls) -> list[str]:
        """Get list of supported vectorization methods.

        Returns:
            List of supported method names.
        """
        return list(cls._VECTORIZERS.keys())
