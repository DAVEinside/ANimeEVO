"""
Genetic algorithm implementation for anime character evolution.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging

from ..attributes.character_attributes import CharacterAttributes

# Setup logging
logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    """
    Implementation of genetic algorithm for character evolution.
    Handles selection, crossover, and mutation operations.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_size: int = 2,
        selection_method: str = "tournament",
        tournament_size: int = 3
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)
            elite_size: Number of top individuals to preserve unchanged
            selection_method: Method for parent selection ("tournament", "roulette", "rank")
            tournament_size: Size of tournament when using tournament selection
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.selection_method = selection_method
        self.tournament_size = tournament_size
    
    def select_parents(
        self, 
        population: List[CharacterAttributes]
    ) -> List[Tuple[CharacterAttributes, CharacterAttributes]]:
        """
        Select parent pairs for breeding the next generation.
        
        Args:
            population: Current population of characters
            
        Returns:
            List of parent pairs (tuples)
        """
        # Calculate how many parent pairs we need
        # We need (population_size - elite_size) / 2 rounded up
        num_offspring = len(population) - self.elite_size
        num_pairs = (num_offspring + 1) // 2  # Round up
        
        # Apply the selected selection method
        if self.selection_method == "tournament":
            return self._tournament_selection(population, num_pairs)
        elif self.selection_method == "roulette":
            return self._roulette_selection(population, num_pairs)
        elif self.selection_method == "rank":
            return self._rank_selection(population, num_pairs)
        else:
            logger.warning(f"Unknown selection method: {self.selection_method}. Using tournament selection.")
            return self._tournament_selection(population, num_pairs)
    
    def _tournament_selection(
        self, 
        population: List[CharacterAttributes], 
        num_pairs: int
    ) -> List[Tuple[CharacterAttributes, CharacterAttributes]]:
        """
        Tournament selection method for parent selection.
        
        Args:
            population: Current population
            num_pairs: Number of parent pairs to select
            
        Returns:
            List of parent pairs
        """
        parent_pairs = []
        
        for _ in range(num_pairs):
            # Select parents through tournaments
            parent1 = self._select_tournament_winner(population)
            parent2 = self._select_tournament_winner(population)
            
            # Ensure parents are different (if possible)
            attempts = 0
            while parent1.id == parent2.id and attempts < 5 and len(population) > 1:
                parent2 = self._select_tournament_winner(population)
                attempts += 1
                
            parent_pairs.append((parent1, parent2))
            
        return parent_pairs
    
    def _select_tournament_winner(self, population: List[CharacterAttributes]) -> CharacterAttributes:
        """Select a winner from a random tournament."""
        # Select random contestants
        tournament_size = min(self.tournament_size, len(population))
        contestants = random.sample(population, tournament_size)
        
        # Find the contestant with highest fitness
        return max(contestants, key=lambda x: x.fitness_score)
    
    def _roulette_selection(
        self, 
        population: List[CharacterAttributes], 
        num_pairs: int
    ) -> List[Tuple[CharacterAttributes, CharacterAttributes]]:
        """
        Roulette wheel (fitness proportionate) selection.
        
        Args:
            population: Current population
            num_pairs: Number of parent pairs to select
            
        Returns:
            List of parent pairs
        """
        parent_pairs = []
        
        # Calculate fitness sum and probabilities
        fitness_values = [max(0.0001, char.fitness_score) for char in population]  # Ensure non-zero
        fitness_sum = sum(fitness_values)
        
        # Handle case where all fitness values are 0
        if fitness_sum <= 0:
            # Fall back to random selection
            logger.warning("All fitness values are zero or negative. Using random selection.")
            for _ in range(num_pairs):
                if len(population) > 1:
                    parent1, parent2 = random.sample(population, 2)
                else:
                    parent1 = parent2 = population[0]
                parent_pairs.append((parent1, parent2))
            return parent_pairs
        
        probabilities = [fitness / fitness_sum for fitness in fitness_values]
        
        # Select parents using probabilities
        for _ in range(num_pairs):
            # Select two parents with replacement
            parents_idx = np.random.choice(len(population), size=2, p=probabilities, replace=True)
            parent1 = population[parents_idx[0]]
            parent2 = population[parents_idx[1]]
            
            # Try to ensure parents are different if possible
            attempts = 0
            while parent1.id == parent2.id and attempts < 5 and len(population) > 1:
                parent2_idx = np.random.choice(len(population), p=probabilities)
                parent2 = population[parent2_idx]
                attempts += 1
                
            parent_pairs.append((parent1, parent2))
            
        return parent_pairs
    
    def _rank_selection(
        self, 
        population: List[CharacterAttributes], 
        num_pairs: int
    ) -> List[Tuple[CharacterAttributes, CharacterAttributes]]:
        """
        Rank-based selection method.
        
        Args:
            population: Current population
            num_pairs: Number of parent pairs to select
            
        Returns:
            List of parent pairs
        """
        parent_pairs = []
        
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        
        # Assign ranks (higher rank = higher selection probability)
        ranks = list(range(1, len(sorted_pop) + 1))
        ranks.reverse()  # Reverse so highest fitness gets highest rank
        
        # Calculate rank sum for probability calculation
        rank_sum = sum(ranks)
        probabilities = [rank / rank_sum for rank in ranks]
        
        # Select parents using rank probabilities
        for _ in range(num_pairs):
            # Select two parents with replacement
            parents_idx = np.random.choice(len(sorted_pop), size=2, p=probabilities, replace=True)
            parent1 = sorted_pop[parents_idx[0]]
            parent2 = sorted_pop[parents_idx[1]]
            
            # Try to ensure parents are different if possible
            attempts = 0
            while parent1.id == parent2.id and attempts < 5 and len(sorted_pop) > 1:
                parent2_idx = np.random.choice(len(sorted_pop), p=probabilities)
                parent2 = sorted_pop[parent2_idx]
                attempts += 1
                
            parent_pairs.append((parent1, parent2))
            
        return parent_pairs
    
    def get_elite(self, population: List[CharacterAttributes]) -> List[CharacterAttributes]:
        """
        Get the elite individuals to preserve in the next generation.
        
        Args:
            population: Current population
            
        Returns:
            List of elite individuals
        """
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        
        # Get the top individuals based on elite_size
        elite_size = min(self.elite_size, len(population))
        elite = sorted_pop[:elite_size]
        
        # Create copies to avoid reference issues
        elite_copies = [character.clone() for character in elite]
        
        # Mark as next generation
        for character in elite_copies:
            character.generation += 1
            # Keep parent IDs the same (they were already set)
            
        return elite_copies
    
    def calculate_population_diversity(self, population: List[CharacterAttributes]) -> float:
        """
        Calculate the diversity of the population.
        
        Args:
            population: Current population
            
        Returns:
            Diversity score (0-1)
        """
        if len(population) <= 1:
            return 0.0
            
        # Extract key attributes for diversity calculation
        attributes_to_check = ['hair_color', 'eye_color', 'personality', 
                              'archetype', 'anime_style', 'art_style']
        
        diversity_scores = []
        
        # Check diversity of each attribute
        for attr in attributes_to_check:
            try:
                # Get all values for this attribute
                if attr in ['personality', 'distinctive_features']:
                    # For list attributes, flatten the lists
                    all_values = []
                    for character in population:
                        values = getattr(character, attr, [])
                        if values:
                            all_values.extend(values)
                else:
                    # For single value attributes
                    all_values = [getattr(character, attr, "") for character in population]
                    all_values = [v for v in all_values if v]  # Remove empty values
                
                if not all_values:
                    continue
                    
                # Calculate unique ratio
                unique_ratio = len(set(all_values)) / len(all_values)
                diversity_scores.append(unique_ratio)
            except Exception as e:
                logger.error(f"Error calculating diversity for attribute {attr}: {e}")
        
        # Return average diversity or 0 if no scores
        return np.mean(diversity_scores) if diversity_scores else 0.0