"""
Trait inheritance system for handling anime character attributes during evolution.
Defines how traits are combined and inherited during character crossover.
"""

import random
import uuid
from typing import List, Dict, Any, Optional, Set, Union
from datetime import datetime
import logging

from ..attributes.character_attributes import CharacterAttributes

# Setup logging
logger = logging.getLogger(__name__)

class TraitInheritance:
    """
    Handles the inheritance of traits during character evolution.
    Implements various inheritance patterns for different attribute types.
    """
    
    def __init__(self):
        """Initialize the trait inheritance system."""
        # Define inheritance patterns for different attribute types
        self.inheritance_patterns = {
            # Simple attributes (inherit from either parent)
            'simple': [
                'gender', 'age_category', 'hair_style', 'height', 
                'body_type', 'skin_tone', 'archetype', 'motivation', 
                'anime_style', 'art_style', 'era'
            ],
            
            # Color attributes (can blend or inherit directly)
            'color': ['hair_color', 'eye_color'],
            
            # List attributes (can combine from both parents)
            'list': [
                'personality', 'distinctive_features', 'color_palette'
            ],
            
            # Text attributes (can concatenate or pick from either parent)
            'text': ['name', 'background'],
            
            # Numeric attributes (can average or inherit directly)
            'numeric': []  # None in the current model
        }
        
        # Define dominance for certain traits (higher value = more dominant)
        self.trait_dominance = {
            'hair_color': {
                'black': 0.9,
                'brown': 0.8,
                'blonde': 0.6,
                'red': 0.7,
                'blue': 0.5,
                'pink': 0.5,
                'purple': 0.5,
                'green': 0.5,
                'white': 0.6
            },
            'eye_color': {
                'brown': 0.9,
                'black': 0.85,
                'blue': 0.7,
                'green': 0.7,
                'red': 0.6,
                'gold': 0.6,
                'purple': 0.6,
                'heterochromia': 0.3
            }
        }
    
    def crossover(self, parent1: CharacterAttributes, parent2: CharacterAttributes) -> CharacterAttributes:
        """
        Create a child character by combining traits from two parents.
        
        Args:
            parent1: First parent character
            parent2: Second parent character
            
        Returns:
            Child character with inherited traits
        """
        # Create a new character instance
        child = CharacterAttributes()
        
        # Set basic evolution metadata
        child.id = str(uuid.uuid4())
        child.creation_date = datetime.now().isoformat()
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.parent_ids = [parent1.id, parent2.id]
        
        # Reset fitness and rating
        child.fitness_score = 0.0
        child.user_rating = 0
        
        # Inherit traits according to patterns
        self._inherit_simple_traits(child, parent1, parent2)
        self._inherit_color_traits(child, parent1, parent2)
        self._inherit_list_traits(child, parent1, parent2)
        self._inherit_text_traits(child, parent1, parent2)
        
        # Custom handling for name
        self._inherit_name(child, parent1, parent2)
        
        return child
    
    def _inherit_simple_traits(self, child: CharacterAttributes, parent1: CharacterAttributes, parent2: CharacterAttributes):
        """Inherit simple traits that come from either parent."""
        for trait in self.inheritance_patterns['simple']:
            # Randomly choose which parent to inherit from
            if random.random() < 0.5:
                setattr(child, trait, getattr(parent1, trait))
            else:
                setattr(child, trait, getattr(parent2, trait))
    
    def _inherit_color_traits(self, child: CharacterAttributes, parent1: CharacterAttributes, parent2: CharacterAttributes):
        """Inherit color traits using dominance or blending."""
        for trait in self.inheritance_patterns['color']:
            # Get values from parents
            value1 = getattr(parent1, trait)
            value2 = getattr(parent2, trait)
            
            # If one parent has no value, use the other
            if not value1:
                setattr(child, trait, value2)
                continue
            if not value2:
                setattr(child, trait, value1)
                continue
            
            # Check if we should use dominance model
            dominance_map = self.trait_dominance.get(trait)
            if dominance_map and random.random() < 0.7:  # 70% chance to use dominance
                # Calculate dominance scores
                dom1 = dominance_map.get(value1, 0.5)
                dom2 = dominance_map.get(value2, 0.5)
                
                # Add randomness
                dom1 += random.uniform(-0.2, 0.2)
                dom2 += random.uniform(-0.2, 0.2)
                
                # Select based on dominance
                if dom1 > dom2:
                    setattr(child, trait, value1)
                else:
                    setattr(child, trait, value2)
            else:
                # Random selection (50/50)
                if random.random() < 0.5:
                    setattr(child, trait, value1)
                else:
                    setattr(child, trait, value2)
    
    def _inherit_list_traits(self, child: CharacterAttributes, parent1: CharacterAttributes, parent2: CharacterAttributes):
        """Inherit list traits by combining elements from both parents."""
        for trait in self.inheritance_patterns['list']:
            # Get values from parents
            list1 = getattr(parent1, trait, []) or []
            list2 = getattr(parent2, trait, []) or []
            
            # Convert to sets for easier manipulation
            set1 = set(list1)
            set2 = set(list2)
            
            # Calculate inheritance approach
            r = random.random()
            result_list = []
            
            if r < 0.3:
                # Approach 1: Common traits (intersection) + some unique ones
                common = set1.intersection(set2)
                unique1 = set1 - set2
                unique2 = set2 - set1
                
                # Add all common traits
                result_list.extend(common)
                
                # Add some unique traits from each parent
                for unique_set in [unique1, unique2]:
                    if unique_set:
                        # Add each unique trait with 30% probability
                        for trait_val in unique_set:
                            if random.random() < 0.3:
                                result_list.append(trait_val)
                
            elif r < 0.6:
                # Approach 2: Some traits from parent 1, some from parent 2
                for trait_val in list1:
                    if random.random() < 0.5:
                        result_list.append(trait_val)
                        
                for trait_val in list2:
                    if trait_val not in result_list and random.random() < 0.5:
                        result_list.append(trait_val)
                        
            else:
                # Approach 3: Most traits from one parent, few from the other
                dominant_list = list1 if random.random() < 0.5 else list2
                recessive_list = list2 if dominant_list is list1 else list1
                
                # Add most from dominant
                for trait_val in dominant_list:
                    if random.random() < 0.8:
                        result_list.append(trait_val)
                        
                # Add few from recessive
                for trait_val in recessive_list:
                    if trait_val not in result_list and random.random() < 0.2:
                        result_list.append(trait_val)
            
            # Limit number of traits to avoid excessive lists
            max_traits = max(len(list1), len(list2))
            if len(result_list) > max_traits:
                result_list = random.sample(result_list, max_traits)
                
            # Set the trait on the child
            setattr(child, trait, result_list)
    
    def _inherit_text_traits(self, child: CharacterAttributes, parent1: CharacterAttributes, parent2: CharacterAttributes):
        """Inherit text traits by selecting or combining text."""
        for trait in self.inheritance_patterns['text']:
            if trait == 'name':
                continue  # Name is handled separately
                
            # Get values from parents
            text1 = getattr(parent1, trait, "")
            text2 = getattr(parent2, trait, "")
            
            # If one parent has no value, use the other
            if not text1:
                setattr(child, trait, text2)
                continue
            if not text2:
                setattr(child, trait, text1)
                continue
                
            # Inheritance approach for text
            r = random.random()
            
            if r < 0.45:
                # Use text from parent 1
                setattr(child, trait, text1)
            elif r < 0.9:
                # Use text from parent 2
                setattr(child, trait, text2)
            else:
                # Combine texts (for short texts only)
                if len(text1) + len(text2) < 200:
                    combined = f"{text1} {text2}"
                    setattr(child, trait, combined)
                else:
                    # If texts are too long, just use one
                    setattr(child, trait, text1 if random.random() < 0.5 else text2)
    
    def _inherit_name(self, child: CharacterAttributes, parent1: CharacterAttributes, parent2: CharacterAttributes):
        """Special handling for character name inheritance."""
        name1 = parent1.name
        name2 = parent2.name
        
        # If both parents have names
        if name1 and name2:
            r = random.random()
            
            if r < 0.4:
                # Use one parent's name
                child.name = name1 if random.random() < 0.5 else name2
            elif r < 0.7:
                # Combine first name from one, last name from other (if applicable)
                name1_parts = name1.split()
                name2_parts = name2.split()
                
                if len(name1_parts) > 1 and len(name2_parts) > 1:
                    # Both have multiple parts - combine first and last
                    if random.random() < 0.5:
                        child.name = f"{name1_parts[0]} {name2_parts[-1]}"
                    else:
                        child.name = f"{name2_parts[0]} {name1_parts[-1]}"
                else:
                    # Can't split names properly, just use one
                    child.name = name1 if random.random() < 0.5 else name2
            elif r < 0.85:
                # Create a portmanteau name
                self._create_portmanteau_name(child, name1, name2)
            else:
                # Leave empty for user to name
                child.name = ""
        elif name1:
            child.name = name1
        elif name2:
            child.name = name2
        else:
            child.name = ""
    
    def _create_portmanteau_name(self, child: CharacterAttributes, name1: str, name2: str):
        """Create a portmanteau (combined) name from two parent names."""
        try:
            # Only use first part of names for simplicity
            name1_first = name1.split()[0]
            name2_first = name2.split()[0]
            
            # Get minimum name length
            min_len = min(len(name1_first), len(name2_first))
            
            # For very short names, just concatenate
            if min_len < 3:
                child.name = name1_first + name2_first
                return
                
            # Try to find a good split point based on vowels
            vowels = "aeiouAEIOU"
            
            # Find vowel positions
            vowel_pos1 = [i for i, c in enumerate(name1_first) if c in vowels]
            vowel_pos2 = [i for i, c in enumerate(name2_first) if c in vowels]
            
            if vowel_pos1 and vowel_pos2:
                # Split at a vowel position
                split1 = random.choice(vowel_pos1)
                split2 = random.choice(vowel_pos2)
                
                # Create portmanteau
                portmanteau = name1_first[:split1+1] + name2_first[split2:]
                child.name = portmanteau.capitalize()
            else:
                # Fallback: split randomly
                split1 = random.randint(1, len(name1_first) - 1)
                child.name = (name1_first[:split1] + name2_first[split1:]).capitalize()
                
        except Exception as e:
            logger.error(f"Error creating portmanteau name: {e}")
            # Fallback to simple name
            child.name = name1 if random.random() < 0.5 else name2