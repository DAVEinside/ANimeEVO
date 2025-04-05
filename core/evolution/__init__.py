"""
Evolution module for the Anime Character Evolution System.

This module provides genetic algorithms for character evolution:
- EvolutionEngine: Main engine for character evolution
- GeneticAlgorithm: Implementation of genetic algorithm operations
- TraitInheritance: System for trait inheritance during evolution
"""

from .evolution_engine import EvolutionEngine, EvolutionRecord
from .genetic_algorithm import GeneticAlgorithm
from .trait_inheritance import TraitInheritance

__all__ = [
    'EvolutionEngine',
    'EvolutionRecord',
    'GeneticAlgorithm',
    'TraitInheritance'
]