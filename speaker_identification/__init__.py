"""
Модуль идентификации спикеров по образцам голоса.
"""

from .embeddings import SpeakerEmbeddingExtractor
from .profiles import SpeakerProfileManager
from .matcher import SpeakerMatcher

__all__ = [
    "SpeakerEmbeddingExtractor",
    "SpeakerProfileManager",
    "SpeakerMatcher",
]
