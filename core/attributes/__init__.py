"""
Attributes module for the Anime Character Evolution System.

This module defines character attributes and styles:
- CharacterAttributes: Class for managing character attributes
- AnimeStyleLibrary: Library of anime and art styles
"""

from .character_attributes import CharacterAttributes
from .anime_styles import AnimeStyleLibrary, AnimeStyle, ArtStyle

__all__ = [
    'CharacterAttributes',
    'AnimeStyleLibrary',
    'AnimeStyle',
    'ArtStyle'
]