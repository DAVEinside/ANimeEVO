"""
Evolution engine for anime character generation and evolution.
Manages the evolution process using genetic algorithms.
"""

import os
import random
import uuid
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime

from ..attributes.character_attributes import CharacterAttributes
from .genetic_algorithm import GeneticAlgorithm
from .trait_inheritance import TraitInheritance

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class EvolutionRecord:
    """Record of an evolution run for tracking lineage and history."""
    id: str
    timestamp: str
    generation_number: int
    parent_ids: List[str]
    character_id: str
    fitness_score: float
    user_rating: int
    mutation_rate: float
    attributes: Dict[str, Any]
    image_path: Optional[str] = None


class EvolutionEngine:
    """
    Main engine for character evolution, coordinating genetic algorithms,
    trait inheritance, and evolution history.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the evolution engine.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize evolution parameters
        self.population_size = self.config['evolution']['population_size']
        self.generations = self.config['evolution']['generations']
        self.mutation_rate = self.config['evolution']['mutation_rate']
        self.crossover_rate = self.config['evolution']['crossover_rate']
        self.elite_size = self.config['evolution']['elite_size']
        self.selection_method = self.config['evolution']['selection_method']
        
        # Initialize genetic algorithm
        self.genetic_algorithm = GeneticAlgorithm(
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elite_size=self.elite_size,
            selection_method=self.selection_method
        )
        
        # Initialize trait inheritance handler
        self.trait_inheritance = TraitInheritance()
        
        # Initialize evolution history
        self.evolution_history: List[EvolutionRecord] = []
        
        # Output directory for saving evolution records
        self.output_dir = self.config['paths']['output_dir']
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Evolution lineage directory
        self.lineage_dir = os.path.join(self.output_dir, "lineage")
        Path(self.lineage_dir).mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")
            logger.warning("Using default configuration.")
            return {
                'evolution': {
                    'population_size': 8,
                    'generations': 5,
                    'mutation_rate': 0.2,
                    'crossover_rate': 0.7,
                    'elite_size': 2,
                    'selection_method': 'tournament'
                },
                'paths': {
                    'output_dir': './outputs',
                }
            }
    
    def initialize_population(self, size: int = None) -> List[CharacterAttributes]:
        """
        Initialize a random population of characters.
        
        Args:
            size: Population size (defaults to config value)
            
        Returns:
            List of random CharacterAttributes
        """
        if size is None:
            size = self.population_size
            
        population = []
        for _ in range(size):
            # Create random character
            character = CharacterAttributes.random(self.config_path)
            population.append(character)
            
        return population
    
    def evolve(
        self, 
        characters: List[CharacterAttributes], 
        fitness_function: Callable[[CharacterAttributes], float] = None,
        generation_count: int = 1
    ) -> List[CharacterAttributes]:
        """
        Evolve a population of characters for multiple generations.
        
        Args:
            characters: Initial population
            fitness_function: Function to calculate fitness (defaults to using stored fitness)
            generation_count: Number of generations to evolve
            
        Returns:
            Evolved population after specified generations
        """
        current_population = characters.copy()
        
        # Use default fitness function if none provided
        if fitness_function is None:
            fitness_function = lambda x: x.fitness_score
        
        for gen in range(generation_count):
            logger.info(f"Evolving generation {gen+1}/{generation_count}")
            
            # Calculate fitness for each character
            for character in current_population:
                if character.fitness_score == 0:  # Only calculate if not already set
                    character.fitness_score = fitness_function(character)
            
            # Sort by fitness for logging
            sorted_pop = sorted(current_population, key=lambda x: x.fitness_score, reverse=True)
            logger.info(f"Best fitness: {sorted_pop[0].fitness_score}, Avg fitness: {np.mean([c.fitness_score for c in current_population])}")
            
            # Create next generation
            current_population = self.evolve_generation(current_population)
            
            # Record evolution history for this generation
            self._record_generation(current_population, gen+1)
            
        return current_population
    
    def evolve_generation(self, population: List[CharacterAttributes]) -> List[CharacterAttributes]:
        """
        Evolve a single generation using genetic algorithm.
        
        Args:
            population: Current population
            
        Returns:
            Next generation population
        """
        # Use genetic algorithm to select parents and create offspring
        parent_pairs = self.genetic_algorithm.select_parents(population)
        
        # Create new population
        new_population = []
        
        # Add elite individuals directly
        elite = self.genetic_algorithm.get_elite(population)
        new_population.extend(elite)
        
        # Create offspring until we reach population size
        while len(new_population) < len(population):
            # Get next parent pair
            if parent_pairs:
                parent1, parent2 = parent_pairs.pop(0)
                
                # Crossover to create child
                child = self.trait_inheritance.crossover(parent1, parent2)
                
                # Mutate with probability
                if random.random() < self.mutation_rate:
                    child = child.mutate(self.mutation_rate)
                    
                # Add to new population
                new_population.append(child)
            else:
                # If we run out of parent pairs but need more offspring,
                # create a mutated version of a random elite individual
                parent = random.choice(elite)
                child = parent.mutate(self.mutation_rate * 1.5)  # Higher mutation rate for diversity
                new_population.append(child)
        
        # Ensure population size remains constant
        if len(new_population) > len(population):
            new_population = new_population[:len(population)]
            
        return new_population
    
    def evolve_single(self, character: CharacterAttributes) -> CharacterAttributes:
        """
        Create a single evolved version of a character (useful for UI).
        
        Args:
            character: Character to evolve
            
        Returns:
            Evolved character
        """
        # Create a simple mutated version
        evolved = character.mutate(self.mutation_rate)
        
        # Record in history
        self._record_evolution(evolved, [character.id])
        
        return evolved
    
    def evolve_pair(self, character1: CharacterAttributes, character2: CharacterAttributes) -> CharacterAttributes:
        """
        Create a child character from two parent characters.
        
        Args:
            character1: First parent
            character2: Second parent
            
        Returns:
            Child character
        """
        # Create child through crossover
        child = self.trait_inheritance.crossover(character1, character2)
        
        # Apply mutation with configured probability
        if random.random() < self.mutation_rate:
            child = child.mutate(self.mutation_rate)
            
        # Record in history
        self._record_evolution(child, [character1.id, character2.id])
        
        return child
    
    def evolve_with_feedback(
        self, 
        characters: List[CharacterAttributes], 
        user_ratings: Dict[str, int],
        feedback_weight: float = 0.7
    ) -> List[CharacterAttributes]:
        """
        Evolve characters based on user feedback ratings.
        
        Args:
            characters: Current population
            user_ratings: Dictionary mapping character IDs to ratings (1-5)
            feedback_weight: Weight of user feedback in fitness calculation
            
        Returns:
            Next generation of characters
        """
        # Update character fitness based on user ratings
        for character in characters:
            if character.id in user_ratings:
                # Set user rating
                character.user_rating = user_ratings[character.id]
                
                # Normalize rating to 0-1 scale
                normalized_rating = (character.user_rating - 1) / 4.0  # Assuming 1-5 rating
                
                # Update fitness as weighted combination of previous fitness and user rating
                if character.fitness_score > 0:
                    character.fitness_score = (
                        (1 - feedback_weight) * character.fitness_score + 
                        feedback_weight * normalized_rating
                    )
                else:
                    character.fitness_score = normalized_rating
        
        # Evolve to next generation
        return self.evolve_generation(characters)
    
    def _record_evolution(self, character: CharacterAttributes, parent_ids: List[str], generation_number: int = None):
        """Record an evolution event in history."""
        if generation_number is None:
            generation_number = character.generation
            
        record = EvolutionRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            generation_number=generation_number,
            parent_ids=parent_ids,
            character_id=character.id,
            fitness_score=character.fitness_score,
            user_rating=character.user_rating,
            mutation_rate=self.mutation_rate,
            attributes=character.to_dict(),
            image_path=None  # Will be filled in later when image is saved
        )
        
        # Add to history
        self.evolution_history.append(record)
        
        # Save to file if configured
        if self.config['logging']['save_evolution_history']:
            self._save_evolution_record(record)
    
    def _record_generation(self, population: List[CharacterAttributes], generation_number: int):
        """Record an entire generation in evolution history."""
        for character in population:
            self._record_evolution(character, character.parent_ids, generation_number)
    
    def _save_evolution_record(self, record: EvolutionRecord):
        """Save evolution record to file."""
        try:
            # Create path
            filename = f"evolution_{record.character_id}_{record.id[:8]}.json"
            filepath = os.path.join(self.lineage_dir, filename)
            
            # Convert to dictionary
            record_dict = {
                "id": record.id,
                "timestamp": record.timestamp,
                "generation_number": record.generation_number,
                "parent_ids": record.parent_ids,
                "character_id": record.character_id,
                "fitness_score": record.fitness_score,
                "user_rating": record.user_rating,
                "mutation_rate": record.mutation_rate,
                "attributes": record.attributes,
                "image_path": record.image_path
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(record_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving evolution record: {e}")
    
    def set_image_path(self, character_id: str, image_path: str):
        """
        Set the image path for a character in evolution history.
        Called after image generation.
        
        Args:
            character_id: ID of the character
            image_path: Path to the generated image
        """
        # Update in memory records
        for record in self.evolution_history:
            if record.character_id == character_id:
                record.image_path = image_path
                
                # Re-save the record
                if self.config['logging']['save_evolution_history']:
                    self._save_evolution_record(record)
    
    def get_lineage(self, character_id: str) -> List[Dict[str, Any]]:
        """
        Get the evolutionary lineage of a character.
        
        Args:
            character_id: ID of the character
            
        Returns:
            List of ancestor records in chronological order
        """
        # Build the ancestry tree recursively
        ancestors = []
        self._find_ancestors(character_id, ancestors)
        
        # Sort by generation and timestamp
        ancestors.sort(key=lambda x: (x["generation_number"], x["timestamp"]))
        
        return ancestors
    
    def _find_ancestors(self, character_id: str, result: List[Dict[str, Any]]):
        """Recursively find ancestors of a character."""
        # Find the character record
        character_record = None
        for record in self.evolution_history:
            if record.character_id == character_id:
                character_record = record
                break
                
        if character_record is None:
            # Try loading from file
            character_record = self._load_record_from_file(character_id)
            
        if character_record is None:
            return
            
        # Add to result if not already present
        if not any(r["character_id"] == character_id for r in result):
            result.append({
                "id": character_record.id,
                "character_id": character_record.character_id,
                "generation_number": character_record.generation_number,
                "timestamp": character_record.timestamp,
                "parent_ids": character_record.parent_ids,
                "fitness_score": character_record.fitness_score,
                "user_rating": character_record.user_rating,
                "image_path": character_record.image_path
            })
            
        # Recursively find parents
        for parent_id in character_record.parent_ids:
            self._find_ancestors(parent_id, result)
    
    def _load_record_from_file(self, character_id: str) -> Optional[EvolutionRecord]:
        """Load an evolution record from file."""
        lineage_dir = Path(self.lineage_dir)
        if not lineage_dir.exists():
            return None
            
        # Find files that match the character ID
        matching_files = list(lineage_dir.glob(f"evolution_{character_id}_*.json"))
        
        if not matching_files:
            return None
            
        try:
            # Load the first matching file
            with open(matching_files[0], 'r') as f:
                data = json.load(f)
                
            # Create record from data
            record = EvolutionRecord(
                id=data["id"],
                timestamp=data["timestamp"],
                generation_number=data["generation_number"],
                parent_ids=data["parent_ids"],
                character_id=data["character_id"],
                fitness_score=data["fitness_score"],
                user_rating=data["user_rating"],
                mutation_rate=data["mutation_rate"],
                attributes=data["attributes"],
                image_path=data["image_path"]
            )
            
            return record
        except Exception as e:
            logger.error(f"Error loading evolution record: {e}")
            return None
    
    def save_lineage_visualization(self, character_id: str, output_path: str = None) -> Optional[str]:
        """
        Generate and save a visualization of a character's lineage.
        
        Args:
            character_id: ID of the character
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization file or None if failed
        """
        try:
            from graphviz import Digraph
            
            # Get the lineage
            lineage = self.get_lineage(character_id)
            
            if not lineage:
                logger.warning(f"No lineage found for character {character_id}")
                return None
                
            # Create a directed graph
            dot = Digraph(comment=f'Character {character_id} Lineage')
            
            # Add nodes for each character
            for record in lineage:
                # Create node label
                label = f"ID: {record['character_id'][:8]}\n"
                label += f"Gen: {record['generation_number']}\n"
                label += f"Fitness: {record['fitness_score']:.2f}\n"
                
                # Highlight the target character
                if record['character_id'] == character_id:
                    dot.node(record['character_id'], label, color='red', style='filled', fillcolor='lightyellow')
                else:
                    dot.node(record['character_id'], label)
            
            # Add edges for parent-child relationships
            for record in lineage:
                for parent_id in record['parent_ids']:
                    dot.edge(parent_id, record['character_id'])
            
            # Set default output path if not provided
            if output_path is None:
                output_dir = os.path.join(self.output_dir, "lineage_visualizations")
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_path = os.path.join(output_dir, f"lineage_{character_id}.pdf")
            
            # Render the graph
            dot.render(output_path.replace('.pdf', ''), format='pdf', cleanup=True)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating lineage visualization: {e}")
            return None