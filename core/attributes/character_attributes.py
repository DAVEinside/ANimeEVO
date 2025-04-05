"""
Character attributes system for anime character generation and evolution.
"""

import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Any
import json
import uuid
from datetime import datetime
import logging
import yaml
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CharacterAttributes:
    """
    Dataclass to represent anime character attributes.
    Used for both generation and evolution.
    """
    # Basic identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Basic physical attributes
    gender: str = "female"  # female, male, androgynous, etc.
    age_category: str = "teen"  # child, teen, young adult, adult, elderly
    
    # Appearance
    hair_color: str = "black"
    hair_style: str = ""  # long, short, twin tails, etc.
    eye_color: str = "blue"
    skin_tone: str = "fair"
    height: str = "average"  # short, average, tall
    body_type: str = "average"  # slender, average, muscular, etc.
    distinctive_features: List[str] = field(default_factory=list)  # glasses, scars, etc.
    
    # Personality and character
    personality: List[str] = field(default_factory=list)  # cheerful, serious, etc.
    archetype: str = ""  # tsundere, kuudere, etc.
    motivation: str = ""
    background: str = ""
    
    # Visual style
    anime_style: str = ""  # shonen, shojo, seinen, etc.
    art_style: str = ""  # detailed, minimalist, etc.
    color_palette: List[str] = field(default_factory=list)
    era: str = "modern"  # 80s, 90s, modern, futuristic
    
    # Evolution metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_score: float = 0.0
    user_rating: int = 0
    
    # Extra attributes for flexibility
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values and validate."""
        # Ensure lists are initialized
        if self.personality is None:
            self.personality = []
        if self.distinctive_features is None:
            self.distinctive_features = []
        if self.color_palette is None:
            self.color_palette = []
        if self.parent_ids is None:
            self.parent_ids = []
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterAttributes':
        """Create a CharacterAttributes instance from a dictionary."""
        # Filter out any keys that are not valid attributes
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    @classmethod
    def random(cls, config_path: str = "config/config.yaml") -> 'CharacterAttributes':
        """Create a random character based on configuration options."""
        # Load configuration for random generation
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Fallback to default options
            return cls._random_default()
            
        # Extract character attribute options from config
        try:
            char_config = config.get('character', {})
            attr_options = char_config.get('attributes', {})
            
            # Create a new instance
            char = cls()
            
            # Random gender
            genders = ["female", "male", "androgynous"]
            char.gender = random.choice(genders)
            
            # Random age category
            age_categories = ["child", "teen", "young adult", "adult", "elderly"]
            char.age_category = random.choice(age_categories)
            
            # Get physical attribute options
            physical = attr_options.get('physical', [])
            
            # Random hair color from common anime colors
            hair_colors = ["black", "brown", "blonde", "red", "blue", "pink", "purple", "white", "green"]
            char.hair_color = random.choice(hair_colors)
            
            # Random eye color from common anime colors
            eye_colors = ["blue", "green", "brown", "red", "gold", "purple", "heterochromia"]
            char.eye_color = random.choice(eye_colors)
            
            # Random skin tone
            skin_tones = ["fair", "pale", "tan", "dark"]
            char.skin_tone = random.choice(skin_tones)
            
            # Random body type
            body_types = ["slender", "average", "athletic", "muscular", "petite"]
            char.body_type = random.choice(body_types)
            
            # Random height
            heights = ["short", "average", "tall"]
            char.height = random.choice(heights)
            
            # Random distinctive features (0-2)
            all_features = ["glasses", "scars", "freckles", "tattoo", "birthmark", 
                            "eye patch", "fangs", "pointed ears", "hair ornaments", 
                            "mechanical parts"]
            num_features = random.randint(0, 2)
            char.distinctive_features = random.sample(all_features, num_features)
            
            # Get personality attribute options
            personality_options = attr_options.get('personality', [])
            
            # Random personality traits (1-3)
            all_traits = ["cheerful", "serious", "shy", "outgoing", "brave", "cautious", 
                          "determined", "lazy", "energetic", "calm", "hot-headed", 
                          "logical", "emotional", "creative", "practical", "loyal", 
                          "rebellious", "honest", "mischievous", "disciplined"]
            num_traits = random.randint(1, 3)
            char.personality = random.sample(all_traits, num_traits)
            
            # Random archetype
            archetypes = ["tsundere", "kuudere", "yandere", "dandere", "deredere", 
                          "himedere", "protagonist", "rival", "mentor", "trickster"]
            char.archetype = random.choice(archetypes)
            
            # Get style attribute options
            style_options = attr_options.get('style', [])
            
            # Random anime style
            anime_styles = ["shonen", "shojo", "seinen", "isekai", "mecha", "chibi", 
                            "fantasy", "sci-fi", "slice of life"]
            char.anime_style = random.choice(anime_styles)
            
            # Random art style
            art_styles = ["detailed", "minimalist", "sketch", "watercolor", "vibrant", 
                          "monochrome", "pastel"]
            char.art_style = random.choice(art_styles)
            
            # Random era
            eras = ["80s", "90s", "modern", "futuristic", "historical", "medieval", 
                    "edo period", "post-apocalyptic"]
            char.era = random.choice(eras)
            
            return char
            
        except Exception as e:
            logger.error(f"Error creating random character: {e}")
            return cls._random_default()
    
    @classmethod
    def _random_default(cls) -> 'CharacterAttributes':
        """Create a random character with default options."""
        char = cls()
        
        # Random basic attributes
        char.gender = random.choice(["female", "male"])
        char.age_category = random.choice(["teen", "young adult"])
        char.hair_color = random.choice(["black", "brown", "blonde", "red", "blue"])
        char.eye_color = random.choice(["blue", "green", "brown", "red"])
        
        # Random personality (1-2 traits)
        all_traits = ["cheerful", "serious", "shy", "outgoing", "brave", "cautious"]
        num_traits = random.randint(1, 2)
        char.personality = random.sample(all_traits, num_traits)
        
        # Random anime style
        char.anime_style = random.choice(["shonen", "shojo", "seinen"])
        
        return char
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CharacterAttributes':
        """Create a CharacterAttributes instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save(self, directory: str = "data/character_templates") -> str:
        """
        Save character attributes to a JSON file.
        
        Args:
            directory: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Generate filename from ID and name
        name_part = self.name.lower().replace(" ", "_") if self.name else "character"
        filename = f"{name_part}_{self.id[:8]}.json"
        filepath = Path(directory) / filename
        
        # Save to file
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        
        return str(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'CharacterAttributes':
        """
        Load character attributes from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            CharacterAttributes instance
        """
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())
    
    def clone(self) -> 'CharacterAttributes':
        """Create a clone of this character."""
        return CharacterAttributes.from_dict(self.to_dict())
    
    def mutate(self, mutation_rate: float = 0.3) -> 'CharacterAttributes':
        """
        Create a mutated version of this character.
        
        Args:
            mutation_rate: Probability of each attribute mutating (0-1)
            
        Returns:
            Mutated CharacterAttributes instance
        """
        # Clone the character
        mutated = self.clone()
        
        # Set as next generation
        mutated.generation = self.generation + 1
        mutated.id = str(uuid.uuid4())
        mutated.parent_ids = [self.id]
        mutated.creation_date = datetime.now().isoformat()
        
        # Reset ratings
        mutated.fitness_score = 0.0
        mutated.user_rating = 0
        
        # Randomly mutate attributes
        # Hair color
        if random.random() < mutation_rate:
            hair_colors = ["black", "brown", "blonde", "red", "blue", "pink", "purple", "white", "green"]
            mutated.hair_color = random.choice(hair_colors)
            
        # Eye color
        if random.random() < mutation_rate:
            eye_colors = ["blue", "green", "brown", "red", "gold", "purple", "heterochromia"]
            mutated.eye_color = random.choice(eye_colors)
            
        # Personality - add or remove a trait
        if random.random() < mutation_rate:
            all_traits = ["cheerful", "serious", "shy", "outgoing", "brave", "cautious", 
                          "determined", "lazy", "energetic", "calm", "hot-headed", 
                          "logical", "emotional", "creative"]
            
            if random.random() < 0.5 and mutated.personality:
                # Remove a trait
                mutated.personality.pop(random.randrange(len(mutated.personality)))
            else:
                # Add a trait
                new_traits = [t for t in all_traits if t not in mutated.personality]
                if new_traits:
                    mutated.personality.append(random.choice(new_traits))
        
        # Distinctive features - add or remove a feature
        if random.random() < mutation_rate:
            all_features = ["glasses", "scars", "freckles", "tattoo", "birthmark", 
                           "eye patch", "fangs", "pointed ears", "hair ornaments"]
            
            if random.random() < 0.5 and mutated.distinctive_features:
                # Remove a feature
                mutated.distinctive_features.pop(random.randrange(len(mutated.distinctive_features)))
            else:
                # Add a feature
                new_features = [f for f in all_features if f not in mutated.distinctive_features]
                if new_features:
                    mutated.distinctive_features.append(random.choice(new_features))
        
        # Art style
        if random.random() < mutation_rate:
            art_styles = ["detailed", "minimalist", "sketch", "watercolor", "vibrant", 
                         "monochrome", "pastel"]
            mutated.art_style = random.choice(art_styles)
            
        return mutated

def fields(cls):
    """Get all fields from a dataclass."""
    import dataclasses
    return dataclasses.fields(cls)