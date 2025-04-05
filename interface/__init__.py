"""
Interface package for the Anime Character Evolution System.

This package contains user interface components:
- web_app: Web-based interface using Flask
- cli: Command-line interface
"""

from .web_app import AnimeEvolutionWebApp
from .cli import AnimeEvolutionCLI

__all__ = [
    'AnimeEvolutionWebApp',
    'AnimeEvolutionCLI'
]