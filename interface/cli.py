"""
Command-line interface for the anime character evolution system.
Provides a text-based interface for character creation, evolution, and visualization.
"""

import os
import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import random
import time

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core system imports
from core.attributes.character_attributes import CharacterAttributes
from core.attributes.anime_styles import AnimeStyleLibrary
from core.diffusion.anime_pipeline import AnimePipeline
from core.evolution.evolution_engine import EvolutionEngine
from multimodal.image_generator import ImageGenerator
from multimodal.animation_generator import AnimationGenerator
from multimodal.voice_generator import VoiceGenerator

# Setup logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimeEvolutionCLI:
    """Command-line interface for anime character evolution system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the CLI application.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Setup directories
        self.setup_directories()
        
        # Initialize core components
        self.initialize_components()
        
        # Initialize argument parser
        self.parser = self._setup_argument_parser()
    
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
                'interface': {
                    'cli': {
                        'enabled': True
                    }
                },
                'paths': {
                    'output_dir': './outputs',
                }
            }
    
    def setup_directories(self):
        """Setup required directories."""
        # Ensure output directory exists
        self.output_dir = self.config['paths']['output_dir']
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create character directory
        self.character_dir = os.path.join(self.output_dir, "characters")
        Path(self.character_dir).mkdir(parents=True, exist_ok=True)
        
        # Create image directory
        self.image_dir = os.path.join(self.output_dir, "images")
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)
    
    def initialize_components(self):
        """Initialize core system components."""
        logger.info("Initializing system components...")
        
        # Initialize anime style library
        self.style_library = AnimeStyleLibrary(config_path=self.config_path)
        
        # Initialize anime pipeline for image generation
        self.pipeline = AnimePipeline(config_path=self.config_path)
        
        # Initialize evolution engine
        self.evolution_engine = EvolutionEngine(config_path=self.config_path)
        
        # Initialize generators
        self.image_generator = ImageGenerator(
            config_path=self.config_path,
            pipeline=self.pipeline
        )
        
        self.animation_generator = AnimationGenerator(
            config_path=self.config_path
        )
        
        self.voice_generator = VoiceGenerator(
            config_path=self.config_path
        )
        
        logger.info("System components initialized")
    
    def _setup_argument_parser(self) -> argparse.ArgumentParser:
        """Setup the argument parser for CLI commands."""
        parser = argparse.ArgumentParser(
            description="Anime Character Evolution System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  Create a random character:
    python cli.py create --random
            
  Create a character with specific attributes:
    python cli.py create --name "Sakura" --gender female --hair-color pink --eye-color green --anime-style shojo
            
  Evolve a character:
    python cli.py evolve --character-id abcd1234
            
  Create a character sheet:
    python cli.py sheet --character-id abcd1234
            
  Generate an animation:
    python cli.py animate --character-id abcd1234
            """
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Create command
        create_parser = subparsers.add_parser("create", help="Create a new character")
        create_parser.add_argument("--random", action="store_true", help="Create a random character")
        create_parser.add_argument("--name", help="Character name")
        create_parser.add_argument("--gender", choices=["female", "male", "androgynous"], help="Character gender")
        create_parser.add_argument("--age", choices=["child", "teen", "young adult", "adult", "elderly"], help="Character age category")
        create_parser.add_argument("--hair-color", help="Hair color")
        create_parser.add_argument("--eye-color", help="Eye color")
        create_parser.add_argument("--anime-style", help="Anime style")
        create_parser.add_argument("--art-style", help="Art style")
        create_parser.add_argument("--personality", nargs='+', help="Personality traits (space-separated)")
        create_parser.add_argument("--features", nargs='+', help="Distinctive features (space-separated)")
        create_parser.add_argument("--prompt", help="Custom prompt for generation")
        create_parser.add_argument("--negative-prompt", help="Negative prompt for generation")
        create_parser.add_argument("--samples", type=int, default=1, help="Number of image samples to generate")
        create_parser.add_argument("--output-file", "-o", help="Output file to save character data")
        
        # List command
        list_parser = subparsers.add_parser("list", help="List characters")
        list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of characters to list")
        list_parser.add_argument("--sort", choices=["name", "date", "generation"], default="date", help="Sort characters by field")
        
        # Show command
        show_parser = subparsers.add_parser("show", help="Show character details")
        show_parser.add_argument("--character-id", "-id", required=True, help="Character ID to show")
        show_parser.add_argument("--show-image", action="store_true", help="Show character image (if supported)")
        
        # Evolve command
        evolve_parser = subparsers.add_parser("evolve", help="Evolve a character")
        evolve_parser.add_argument("--character-id", "-id", required=True, help="Character ID to evolve")
        evolve_parser.add_argument("--type", choices=["single", "pair", "generation"], default="single", help="Evolution type")
        evolve_parser.add_argument("--other-id", help="Second character ID for pair evolution")
        evolve_parser.add_argument("--population-size", type=int, default=4, help="Population size for generation evolution")
        evolve_parser.add_argument("--mutation-rate", type=float, help="Mutation rate (0-1)")
        evolve_parser.add_argument("--output-file", "-o", help="Output file to save evolved character data")
        
        # Generate command
        generate_parser = subparsers.add_parser("generate", help="Generate character images")
        generate_parser.add_argument("--character-id", "-id", required=True, help="Character ID to generate images for")
        generate_parser.add_argument("--samples", type=int, default=1, help="Number of image samples to generate")
        generate_parser.add_argument("--style", help="Apply specific anime style")
        generate_parser.add_argument("--prompt", help="Custom prompt for generation")
        generate_parser.add_argument("--output-dir", "-o", help="Output directory to save images")
        
        # Sheet command
        sheet_parser = subparsers.add_parser("sheet", help="Create a character sheet")
        sheet_parser.add_argument("--character-id", "-id", required=True, help="Character ID for the character sheet")
        sheet_parser.add_argument("--include-info", action="store_true", default=True, help="Include character info on sheet")
        sheet_parser.add_argument("--output-file", "-o", help="Output file to save the character sheet")
        
        # Animate command
        animate_parser = subparsers.add_parser("animate", help="Generate character animation")
        animate_parser.add_argument("--character-id", "-id", required=True, help="Character ID to animate")
        animate_parser.add_argument("--duration", type=float, default=3.0, help="Animation duration in seconds")
        animate_parser.add_argument("--fps", type=int, default=24, help="Frames per second")
        animate_parser.add_argument("--motion", choices=["default", "breathing", "pan", "zoom"], default="default", help="Motion type")
        animate_parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", help="Output format")
        animate_parser.add_argument("--output-file", "-o", help="Output file to save animation")
        
        # Voice command
        voice_parser = subparsers.add_parser("voice", help="Generate character voice")
        voice_parser.add_argument("--character-id", "-id", required=True, help="Character ID to generate voice for")
        voice_parser.add_argument("--text", required=True, help="Text to synthesize")
        voice_parser.add_argument("--preset", help="Voice preset to use")
        voice_parser.add_argument("--format", choices=["wav", "mp3"], default="wav", help="Output format")
        voice_parser.add_argument("--output-file", "-o", help="Output file to save voice")
        
        # Lineage command
        lineage_parser = subparsers.add_parser("lineage", help="Show character lineage")
        lineage_parser.add_argument("--character-id", "-id", required=True, help="Character ID to show lineage for")
        lineage_parser.add_argument("--visualize", action="store_true", help="Generate lineage visualization")
        lineage_parser.add_argument("--output-file", "-o", help="Output file to save lineage visualization")
        
        return parser
    
    def run(self):
        """Run the CLI application."""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
            
        if args.command == "create":
            self.command_create(args)
        elif args.command == "list":
            self.command_list(args)
        elif args.command == "show":
            self.command_show(args)
        elif args.command == "evolve":
            self.command_evolve(args)
        elif args.command == "generate":
            self.command_generate(args)
        elif args.command == "sheet":
            self.command_sheet(args)
        elif args.command == "animate":
            self.command_animate(args)
        elif args.command == "voice":
            self.command_voice(args)
        elif args.command == "lineage":
            self.command_lineage(args)
        else:
            logger.error(f"Unknown command: {args.command}")
    
    def command_create(self, args):
        """Create a new character."""
        if args.random:
            logger.info("Creating random character...")
            character = CharacterAttributes.random(self.config_path)
        else:
            logger.info("Creating character with specified attributes...")
            character = CharacterAttributes()
            
            # Set attributes from arguments
            if args.name:
                character.name = args.name
            if args.gender:
                character.gender = args.gender
            if args.age:
                character.age_category = args.age
            if args.hair_color:
                character.hair_color = args.hair_color
            if args.eye_color:
                character.eye_color = args.eye_color
            if args.anime_style:
                character.anime_style = args.anime_style
            if args.art_style:
                character.art_style = args.art_style
            if args.personality:
                character.personality = args.personality
            if args.features:
                character.distinctive_features = args.features
        
        # Display character info
        print("\n===== CHARACTER CREATED =====")
        print(f"ID: {character.id}")
        print(f"Name: {character.name or 'Unnamed'}")
        print(f"Gender: {character.gender}")
        print(f"Age: {character.age_category}")
        print(f"Hair: {character.hair_color}")
        print(f"Eyes: {character.eye_color}")
        print(f"Style: {character.anime_style or 'Default'}")
        
        # Generate images
        print("\nGenerating character images...")
        
        custom_prompt = args.prompt if hasattr(args, 'prompt') and args.prompt else None
        negative_prompt = args.negative_prompt if hasattr(args, 'negative_prompt') and args.negative_prompt else None
        
        # Show a simple progress indicator
        for _ in range(5):
            print(".", end="", flush=True)
            time.sleep(0.5)
        print()
        
        images, output = self.image_generator.generate_images(
            attributes=character,
            num_samples=args.samples,
            custom_prompt=custom_prompt,
            custom_negative_prompt=negative_prompt
        )
        
        # Save images
        image_paths = self.image_generator.save_images(
            images=images,
            character=character,
            output=output,
            prefix="cli"
        )
        
        print(f"\nGenerated {len(image_paths)} images:")
        for i, path in enumerate(image_paths):
            print(f"  Image {i+1}: {path}")
        
        # Save character data
        if args.output_file:
            output_path = args.output_file
        else:
            output_path = os.path.join(self.character_dir, f"{character.id}.json")
            
        with open(output_path, 'w') as f:
            json.dump(character.to_dict(), f, indent=2)
            
        print(f"\nCharacter data saved to: {output_path}")
        
        # Try to display the first image if supported
        try:
            if len(image_paths) > 0 and os.name != 'nt':  # Skip on Windows
                if hasattr(Image, 'show') and not os.environ.get('SSH_CONNECTION'):
                    print("\nDisplaying image (close window to continue)...")
                    Image.open(image_paths[0]).show()
        except Exception as e:
            logger.debug(f"Could not display image: {e}")
    
    def command_list(self, args):
        """List available characters."""
        characters = []
        
        # Find all character JSON files
        try:
            for file_path in Path(self.character_dir).glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    characters.append({
                        'id': data.get('id', ''),
                        'name': data.get('name', 'Unnamed'),
                        'gender': data.get('gender', ''),
                        'age_category': data.get('age_category', ''),
                        'anime_style': data.get('anime_style', ''),
                        'generation': data.get('generation', 0),
                        'creation_date': data.get('creation_date', ''),
                        'file_path': str(file_path)
                    })
                except Exception as e:
                    logger.warning(f"Error reading character file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error listing characters: {e}")
        
        # Sort characters
        if args.sort == "name":
            characters.sort(key=lambda x: x.get('name', '').lower())
        elif args.sort == "date":
            characters.sort(key=lambda x: x.get('creation_date', ''), reverse=True)
        elif args.sort == "generation":
            characters.sort(key=lambda x: x.get('generation', 0), reverse=True)
        
        # Limit the number of characters
        characters = characters[:args.limit]
        
        # Display characters
        if characters:
            print("\n===== CHARACTER LIST =====")
            print(f"{'ID':<10} {'Name':<20} {'Gender':<10} {'Age':<12} {'Style':<15} {'Gen':<5}")
            print(f"{'-'*10} {'-'*20} {'-'*10} {'-'*12} {'-'*15} {'-'*5}")
            
            for char in characters:
                print(f"{char['id'][:8]:<10} {char['name'][:20]:<20} {char['gender'][:10]:<10} "
                      f"{char['age_category'][:12]:<12} {char['anime_style'][:15]:<15} {char['generation']:<5}")
        else:
            print("\nNo characters found.")
    
    def command_show(self, args):
        """Show character details."""
        character_id = args.character_id
        
        # Find character file
        character_file = None
        for file_path in Path(self.character_dir).glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if data.get('id', '').startswith(character_id):
                    character_file = file_path
                    character_data = data
                    break
            except Exception:
                pass
        
        if character_file is None:
            logger.error(f"Character with ID {character_id} not found.")
            return
        
        # Display character details
        print("\n===== CHARACTER DETAILS =====")
        print(f"ID: {character_data.get('id', '')}")
        print(f"Name: {character_data.get('name', 'Unnamed')}")
        print(f"Gender: {character_data.get('gender', '')}")
        print(f"Age: {character_data.get('age_category', '')}")
        print(f"Hair: {character_data.get('hair_color', '')}")
        print(f"Eyes: {character_data.get('eye_color', '')}")
        print(f"Style: {character_data.get('anime_style', 'Default')}")
        print(f"Art Style: {character_data.get('art_style', 'Default')}")
        print(f"Generation: {character_data.get('generation', 0)}")
        
        # Show personality traits
        personality = character_data.get('personality', [])
        if personality:
            print("\nPersonality Traits:")
            for trait in personality:
                print(f"  - {trait}")
        
        # Show distinctive features
        features = character_data.get('distinctive_features', [])
        if features:
            print("\nDistinctive Features:")
            for feature in features:
                print(f"  - {feature}")
        
        # Show parents if any
        parent_ids = character_data.get('parent_ids', [])
        if parent_ids:
            print("\nParent IDs:")
            for parent_id in parent_ids:
                print(f"  - {parent_id}")
        
        # Show creation date
        creation_date = character_data.get('creation_date', '')
        if creation_date:
            print(f"\nCreation Date: {creation_date}")
        
        # Find and display character image if requested
        if args.show_image:
            # Find image files for this character
            char_id_prefix = character_data.get('id', '')[:8]
            image_paths = []
            
            # Check in subfolders
            for subfolder in Path(self.image_dir).glob("*"):
                if not subfolder.is_dir():
                    continue
                    
                for img_ext in ['png', 'jpg', 'jpeg']:
                    image_paths.extend(list(subfolder.glob(f"*{char_id_prefix}*.{img_ext}")))
            
            if image_paths:
                # Sort by modification time (newest first)
                image_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                # Display first image
                try:
                    if os.name != 'nt':  # Skip on Windows
                        if hasattr(Image, 'show') and not os.environ.get('SSH_CONNECTION'):
                            print("\nDisplaying image (close window to continue)...")
                            Image.open(image_paths[0]).show()
                        else:
                            print(f"\nImage Path: {image_paths[0]}")
                    else:
                        print(f"\nImage Path: {image_paths[0]}")
                except Exception as e:
                    logger.debug(f"Could not display image: {e}")
            else:
                print("\nNo images found for this character.")
    
    def command_evolve(self, args):
        """Evolve a character."""
        character_id = args.character_id
        evolution_type = args.type
        
        # Load character
        character = self._load_character(character_id)
        if character is None:
            logger.error(f"Character with ID {character_id} not found.")
            return
        
        print(f"\n===== EVOLVING CHARACTER: {character.name or 'Unnamed'} =====")
        
        # Set mutation rate if provided
        mutation_rate = None
        if hasattr(args, 'mutation_rate') and args.mutation_rate is not None:
            mutation_rate = args.mutation_rate
            self.evolution_engine.mutation_rate = mutation_rate
            print(f"Using custom mutation rate: {mutation_rate}")
        
        # Handle different evolution types
        if evolution_type == "single":
            # Evolve a single character
            print("Evolving character (single mutation)...")
            
            # Show a simple progress indicator
            for _ in range(3):
                print(".", end="", flush=True)
                time.sleep(0.5)
            print()
            
            evolved = self.evolution_engine.evolve_single(character)
            
            # Display evolution result
            print("\n===== EVOLUTION COMPLETE =====")
            print(f"Original ID: {character.id}")
            print(f"Evolved ID: {evolved.id}")
            print(f"Generation: {evolved.generation}")
            
            # Generate image for evolved character
            print("\nGenerating image for evolved character...")
            
            # Show a simple progress indicator
            for _ in range(5):
                print(".", end="", flush=True)
                time.sleep(0.5)
            print()
            
            images, output = self.image_generator.generate_images(
                attributes=evolved,
                num_samples=1
            )
            
            # Save images
            image_paths = self.image_generator.save_images(
                images=images,
                character=evolved,
                output=output,
                prefix="evolved"
            )
            
            print(f"\nGenerated image: {image_paths[0] if image_paths else 'None'}")
            
            # Save evolved character
            if args.output_file:
                output_path = args.output_file
            else:
                output_path = os.path.join(self.character_dir, f"{evolved.id}.json")
                
            with open(output_path, 'w') as f:
                json.dump(evolved.to_dict(), f, indent=2)
                
            print(f"Evolved character saved to: {output_path}")
            
            # Try to display the image if supported
            try:
                if len(image_paths) > 0 and os.name != 'nt':
                    if hasattr(Image, 'show') and not os.environ.get('SSH_CONNECTION'):
                        print("\nDisplaying image (close window to continue)...")
                        Image.open(image_paths[0]).show()
            except Exception as e:
                logger.debug(f"Could not display image: {e}")
                
        elif evolution_type == "pair":
            # Evolve with another character
            other_id = args.other_id
            if not other_id:
                logger.error("Second character ID (--other-id) is required for pair evolution.")
                return
                
            other_character = self._load_character(other_id)
            if other_character is None:
                logger.error(f"Character with ID {other_id} not found.")
                return
                
            print(f"Evolving character with {other_character.name or 'Unnamed'} (ID: {other_id})...")
            
            # Show a simple progress indicator
            for _ in range(3):
                print(".", end="", flush=True)
                time.sleep(0.5)
            print()
            
            evolved = self.evolution_engine.evolve_pair(character, other_character)
            
            # Display evolution result
            print("\n===== EVOLUTION COMPLETE =====")
            print(f"Parent 1 ID: {character.id}")
            print(f"Parent 2 ID: {other_character.id}")
            print(f"Child ID: {evolved.id}")
            print(f"Generation: {evolved.generation}")
            
            # Generate image for evolved character
            print("\nGenerating image for evolved character...")
            
            # Show a simple progress indicator
            for _ in range(5):
                print(".", end="", flush=True)
                time.sleep(0.5)
            print()
            
            images, output = self.image_generator.generate_images(
                attributes=evolved,
                num_samples=1
            )
            
            # Save images
            image_paths = self.image_generator.save_images(
                images=images,
                character=evolved,
                output=output,
                prefix="evolved"
            )
            
            print(f"\nGenerated image: {image_paths[0] if image_paths else 'None'}")
            
            # Save evolved character
            if args.output_file:
                output_path = args.output_file
            else:
                output_path = os.path.join(self.character_dir, f"{evolved.id}.json")
                
            with open(output_path, 'w') as f:
                json.dump(evolved.to_dict(), f, indent=2)
                
            print(f"Evolved character saved to: {output_path}")
            
            # Try to display the image if supported
            try:
                if len(image_paths) > 0 and os.name != 'nt':
                    if hasattr(Image, 'show') and not os.environ.get('SSH_CONNECTION'):
                        print("\nDisplaying image (close window to continue)...")
                        Image.open(image_paths[0]).show()
            except Exception as e:
                logger.debug(f"Could not display image: {e}")
                
        elif evolution_type == "generation":
            # Evolve a whole generation
            population_size = args.population_size
            print(f"Evolving population of {population_size} characters...")
            
            # Create a small population with the character and some variations
            population = [character]
            
            # Add mutations to fill population
            for _ in range(population_size - 1):
                population.append(character.mutate(self.evolution_engine.mutation_rate))
                
            # Evolve the population
            print("\nPerforming evolution...")
            
            # Show a simple progress indicator
            for _ in range(5):
                print(".", end="", flush=True)
                time.sleep(1)
            print()
            
            evolved_characters = self.evolution_engine.evolve_generation(population)
            
            print("\n===== EVOLUTION COMPLETE =====")
            print(f"Generated {len(evolved_characters)} evolved characters")
            
            # Generate and save evolved characters
            for i, evolved in enumerate(evolved_characters):
                print(f"\n--- Evolved Character {i+1} ---")
                print(f"ID: {evolved.id}")
                print(f"Generation: {evolved.generation}")
                
                # Generate image
                images, output = self.image_generator.generate_images(
                    attributes=evolved,
                    num_samples=1
                )
                
                # Save images
                image_paths = self.image_generator.save_images(
                    images=images,
                    character=evolved,
                    output=output,
                    prefix=f"evolved_gen"
                )
                
                # Save character data
                output_path = os.path.join(self.character_dir, f"{evolved.id}.json")
                with open(output_path, 'w') as f:
                    json.dump(evolved.to_dict(), f, indent=2)
                    
                print(f"Image: {image_paths[0] if image_paths else 'None'}")
                print(f"Saved to: {output_path}")
                
            print("\nEvolution generation complete.")
    
    def command_generate(self, args):
        """Generate character images."""
        character_id = args.character_id
        
        # Load character
        character = self._load_character(character_id)
        if character is None:
            logger.error(f"Character with ID {character_id} not found.")
            return
        
        print(f"\n===== GENERATING IMAGES FOR: {character.name or 'Unnamed'} =====")
        print(f"Generating {args.samples} image samples...")
        
        # Show a simple progress indicator
        for _ in range(5):
            print(".", end="", flush=True)
            time.sleep(0.5)
        print()
        
        # Generate images
        images, output = self.image_generator.generate_images(
            attributes=character,
            num_samples=args.samples,
            custom_prompt=args.prompt if hasattr(args, 'prompt') else None,
            apply_style=args.style if hasattr(args, 'style') else None
        )
        
        # Save images
        if hasattr(args, 'output_dir') and args.output_dir:
            output_dir = args.output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None
            
        image_paths = self.image_generator.save_images(
            images=images,
            character=character,
            output=output,
            prefix="generated"
        )
        
        print(f"\nGenerated {len(image_paths)} images:")
        for i, path in enumerate(image_paths):
            print(f"  Image {i+1}: {path}")
        
        # Try to display the first image if supported
        try:
            if len(image_paths) > 0 and os.name != 'nt':
                if hasattr(Image, 'show') and not os.environ.get('SSH_CONNECTION'):
                    print("\nDisplaying first image (close window to continue)...")
                    Image.open(image_paths[0]).show()
        except Exception as e:
            logger.debug(f"Could not display image: {e}")
    
    def command_sheet(self, args):
        """Create a character sheet."""
        character_id = args.character_id
        
        # Load character
        character = self._load_character(character_id)
        if character is None:
            logger.error(f"Character with ID {character_id} not found.")
            return
        
        print(f"\n===== CREATING CHARACTER SHEET FOR: {character.name or 'Unnamed'} =====")
        
        # Find character images
        char_id_prefix = character.id[:8]
        image_paths = []
        
        # Check in subfolders
        for subfolder in Path(self.image_dir).glob("*"):
            if not subfolder.is_dir():
                continue
                
            for img_ext in ['png', 'jpg', 'jpeg']:
                image_paths.extend(list(subfolder.glob(f"*{char_id_prefix}*.{img_ext}")))
        
        if not image_paths:
            logger.error("No images found for this character. Generating new images...")
            
            # Generate new images
            images, output = self.image_generator.generate_images(
                attributes=character,
                num_samples=4
            )
            
            # Save images
            image_paths = self.image_generator.save_images(
                images=images,
                character=character,
                output=output,
                prefix="sheet"
            )
        
        # Sort by modification time (newest first)
        image_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Limit to 6 images
        image_paths = image_paths[:6]
        
        # Load images
        images = [Image.open(path) for path in image_paths]
        
        print(f"Using {len(images)} images for character sheet...")
        
        # Create character sheet
        output_path = args.output_file if hasattr(args, 'output_file') and args.output_file else None
        
        # Show a simple progress indicator
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.5)
        print()
        
        sheet_path = self.image_generator.create_character_sheet(
            character=character,
            images=images,
            include_info=args.include_info,
            output_path=output_path
        )
        
        print(f"\nCharacter sheet created: {sheet_path}")
        
        # Try to display the sheet if supported
        try:
            if os.name != 'nt':
                if hasattr(Image, 'show') and not os.environ.get('SSH_CONNECTION'):
                    print("\nDisplaying character sheet (close window to continue)...")
                    Image.open(sheet_path).show()
        except Exception as e:
            logger.debug(f"Could not display character sheet: {e}")
    
    def command_animate(self, args):
        """Generate character animation."""
        character_id = args.character_id
        
        # Load character
        character = self._load_character(character_id)
        if character is None:
            logger.error(f"Character with ID {character_id} not found.")
            return
        
        print(f"\n===== GENERATING ANIMATION FOR: {character.name or 'Unnamed'} =====")
        
        # Find a character image to animate
        char_id_prefix = character.id[:8]
        image_paths = []
        
        # Check in subfolders
        for subfolder in Path(self.image_dir).glob("*"):
            if not subfolder.is_dir():
                continue
                
            for img_ext in ['png', 'jpg', 'jpeg']:
                image_paths.extend(list(subfolder.glob(f"*{char_id_prefix}*.{img_ext}")))
        
        reference_image = None
        
        if image_paths:
            # Sort by modification time (newest first)
            image_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            reference_image = Image.open(image_paths[0])
            print(f"Using image: {image_paths[0]}")
        else:
            print("No existing images found. Generating a new image...")
            
            # Generate a new image
            images, _ = self.image_generator.generate_images(
                attributes=character,
                num_samples=1
            )
            
            if images:
                reference_image = images[0]
                print("Generated new reference image.")
            else:
                logger.error("Failed to generate reference image.")
                return
        
        print(f"\nGenerating {args.duration}s animation at {args.fps} FPS...")
        print(f"Motion type: {args.motion}")
        
        # Show a simple progress indicator
        for _ in range(5):
            print(".", end="", flush=True)
            time.sleep(1)
        print()
        
        # Generate animation
        output_path = args.output_file if hasattr(args, 'output_file') and args.output_file else None
        
        animation_path = self.animation_generator.generate_animation(
            character=character,
            reference_image=reference_image,
            duration=args.duration,
            fps=args.fps,
            motion_type=args.motion,
            output_format=args.format,
            output_path=output_path
        )
        
        if animation_path:
            print(f"\nAnimation created: {animation_path}")
        else:
            logger.error("Failed to generate animation.")
    
    def command_voice(self, args):
        """Generate character voice."""
        character_id = args.character_id
        
        # Load character
        character = self._load_character(character_id)
        if character is None:
            logger.error(f"Character with ID {character_id} not found.")
            return
        
        print(f"\n===== GENERATING VOICE FOR: {character.name or 'Unnamed'} =====")
        print(f"Text: \"{args.text}\"")
        
        # Select voice preset
        preset = args.preset
        if not preset:
            # Auto-select based on character attributes
            if character.gender == "female":
                if character.age_category in ["child"]:
                    preset = "child"
                elif character.age_category in ["teen", "young adult"]:
                    preset = "female_teen"
                else:
                    preset = "female_adult"
            elif character.gender == "male":
                if character.age_category in ["child"]:
                    preset = "child"
                elif character.age_category in ["teen", "young adult"]:
                    preset = "male_teen"
                else:
                    preset = "male_adult"
            else:
                preset = "female_teen"  # Default
            
            print(f"Auto-selected voice preset: {preset}")
        else:
            print(f"Using voice preset: {preset}")
        
        # Show a simple progress indicator
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.5)
        print()
        
        # Generate voice
        output_path = args.output_file if hasattr(args, 'output_file') and args.output_file else None
        
        voice_path = self.voice_generator.generate_voice(
            character=character,
            text=args.text,
            preset=preset,
            output_format=args.format,
            output_path=output_path
        )
        
        if voice_path:
            print(f"\nVoice audio created: {voice_path}")
        else:
            logger.error("Failed to generate voice audio.")
    
    def command_lineage(self, args):
        """Show character lineage."""
        character_id = args.character_id
        
        # Load character
        character = self._load_character(character_id)
        if character is None:
            logger.error(f"Character with ID {character_id} not found.")
            return
        
        print(f"\n===== LINEAGE FOR: {character.name or 'Unnamed'} =====")
        
        # Get lineage
        lineage = self.evolution_engine.get_lineage(character_id)
        
        if not lineage:
            print("No lineage found for this character.")
            return
        
        # Display lineage
        print(f"Found {len(lineage)} ancestors/relatives")
        
        # Sort by generation
        lineage.sort(key=lambda x: (x.get("generation_number", 0), x.get("timestamp", "")))
        
        print("\n--- Lineage Tree ---")
        for entry in lineage:
            char_id = entry.get("character_id", "")
            gen = entry.get("generation_number", 0)
            indent = "  " * gen
            
            # Get character name
            char_name = "Unknown"
            char_file = os.path.join(self.character_dir, f"{char_id}.json")
            if os.path.exists(char_file):
                try:
                    with open(char_file, 'r') as f:
                        char_data = json.load(f)
                        char_name = char_data.get("name", "Unnamed")
                except Exception:
                    pass
            
            print(f"{indent}Gen {gen}: {char_name} (ID: {char_id[:8]})")
        
        # Generate visualization if requested
        if args.visualize:
            print("\nGenerating lineage visualization...")
            
            # Show a simple progress indicator
            for _ in range(3):
                print(".", end="", flush=True)
                time.sleep(0.5)
            print()
            
            output_path = args.output_file if hasattr(args, 'output_file') and args.output_file else None
            
            viz_path = self.evolution_engine.save_lineage_visualization(
                character_id=character_id,
                output_path=output_path
            )
            
            if viz_path:
                print(f"Lineage visualization saved to: {viz_path}")
            else:
                logger.error("Failed to generate lineage visualization.")
    
    def _load_character(self, character_id: str) -> Optional[CharacterAttributes]:
        """
        Load a character by ID or partial ID.
        
        Args:
            character_id: Full or partial character ID
            
        Returns:
            CharacterAttributes object or None if not found
        """
        # Try direct file lookup first
        direct_path = os.path.join(self.character_dir, f"{character_id}.json")
        if os.path.exists(direct_path):
            try:
                with open(direct_path, 'r') as f:
                    data = json.load(f)
                return CharacterAttributes.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading character file: {e}")
                return None
        
        # Try partial ID match
        for file_path in Path(self.character_dir).glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if data.get('id', '').startswith(character_id):
                    return CharacterAttributes.from_dict(data)
            except Exception:
                pass
                
        return None


# Main entry point
if __name__ == "__main__":
    # Create and run the CLI application
    cli = AnimeEvolutionCLI()
    cli.run()