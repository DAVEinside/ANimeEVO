"""
Web application interface for the anime character evolution system.
Provides a GUI for character creation, evolution, and visualization.
"""

import os
import sys
import json
import yaml
import base64
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from io import BytesIO
from datetime import datetime

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Flask imports
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, abort

# Image handling
from PIL import Image

# Core system imports
from core.attributes.character_attributes import CharacterAttributes
from core.attributes.anime_styles import AnimeStyleLibrary
from core.diffusion.anime_pipeline import AnimePipeline
from core.evolution.evolution_engine import EvolutionEngine
from multimodal.image_generator import ImageGenerator
from multimodal.animation_generator import AnimationGenerator
from multimodal.voice_generator import VoiceGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimeEvolutionWebApp:
    """Web application for anime character evolution system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the web application.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Setup directories
        self.setup_directories()
        template_dir = os.path.abspath(self.config['interface']['web']['template_dir'])
        static_dir = os.path.abspath(self.config['interface']['web']['static_dir'])
        
        # Initialize Flask application
        self.app = Flask(
            __name__,
            template_folder=template_dir,
            static_folder=static_dir
        )
        
        # Setup routes
        self._setup_routes()
        
        # Initialize core components lazily
        self.pipeline = None
        self.evolution_engine = None
        self.image_generator = None
        self.animation_generator = None
        self.voice_generator = None
        self.style_library = None
        
        # Character cache for UI
        self.character_cache = {}
        
        # Session data
        self.active_characters = []
        self.evolution_history = []
        
        logger.info("Web application initialized")
    
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
                    'web': {
                        'enabled': True,
                        'host': "0.0.0.0",
                        'port': 8080,
                        'template_dir': "./interface/templates",
                        'static_dir': "./interface/static"
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
        
        # Create temporary upload directory
        self.upload_dir = os.path.join(self.output_dir, "uploads")
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        
        # Ensure template directory exists
        template_dir = self.config['interface']['web']['template_dir']
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(template_dir):
            template_dir = os.path.abspath(template_dir)
        Path(template_dir).mkdir(parents=True, exist_ok=True)
        
        # Ensure static directory exists
        static_dir = self.config['interface']['web']['static_dir']
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(static_dir):
            static_dir = os.path.abspath(static_dir)
        Path(static_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_routes(self):
        """Setup Flask routes."""
        # Main pages
        self.app.route("/")(self.index_page)
        self.app.route("/character/create")(self.create_character_page)
        self.app.route("/character/evolve")(self.evolve_character_page)
        self.app.route("/character/view/<character_id>")(self.view_character_page)
        self.app.route("/gallery")(self.gallery_page)
        self.app.route("/settings")(self.settings_page)
        
        # API endpoints
        self.app.route("/api/character/create", methods=["POST"])(self.api_create_character)
        self.app.route("/api/character/random", methods=["GET"])(self.api_random_character)
        self.app.route("/api/character/evolve", methods=["POST"])(self.api_evolve_character)
        self.app.route("/api/character/regenerate", methods=["POST"])(self.api_regenerate_character)
        self.app.route("/api/character/save", methods=["POST"])(self.api_save_character)
        self.app.route("/api/character/<character_id>", methods=["GET"])(self.api_get_character)
        
        self.app.route("/api/styles", methods=["GET"])(self.api_get_styles)
        self.app.route("/api/image/<image_id>", methods=["GET"])(self.api_get_image)
        self.app.route("/api/lineage/<character_id>", methods=["GET"])(self.api_get_lineage)
        
        # Asset routes
        self.app.route("/uploads/<path:filename>")(self.serve_upload)
        self.app.route("/outputs/<path:filename>")(self.serve_output)
    
    def initialize_components(self):
        """Initialize core system components lazily."""
        # Only initialize components when needed
        if self.pipeline is None:
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
    
    def run(self):
        """Run the web application."""
        host = self.config['interface']['web']['host']
        port = self.config['interface']['web']['port']
        
        logger.info(f"Starting web application at http://{host}:{port}")
        
        self.app.run(
            host=host,
            port=port,
            debug=self.config.get('app', {}).get('debug', False)
        )
    
    # Page handlers
    
    def index_page(self):
        """Render the main index page."""
        return render_template(
            "index.html",
            title="Anime Character Evolution System",
            active_characters=self.active_characters
        )
    
    def create_character_page(self):
        """Render the character creation page."""
        # Initialize components if needed
        if self.style_library is None:
            self.initialize_components()
            
        # Get available anime styles
        anime_styles = self.style_library.list_anime_styles()
        art_styles = self.style_library.list_art_styles()
        
        return render_template(
            "character_creation.html",
            title="Create Character",
            anime_styles=anime_styles,
            art_styles=art_styles
        )
    
    def evolve_character_page(self):
        """Render the character evolution page."""
        character_id = request.args.get("id")
        
        # Check if character exists
        if character_id not in self.character_cache:
            return redirect(url_for("create_character_page"))
            
        character = self.character_cache[character_id]
        
        return render_template(
            "evolution_dashboard.html",
            title="Evolve Character",
            character=character,
            character_json=json.dumps(character.to_dict())
        )
    
    def view_character_page(self, character_id):
        """Render the character view page."""
        # Check if character exists
        if character_id not in self.character_cache:
            # Try to load from file
            try:
                character_dir = os.path.join(self.output_dir, "characters")
                character_file = os.path.join(character_dir, f"{character_id}.json")
                
                if os.path.exists(character_file):
                    with open(character_file, 'r') as f:
                        character_data = json.load(f)
                    character = CharacterAttributes.from_dict(character_data)
                    self.character_cache[character_id] = character
                else:
                    return redirect(url_for("gallery_page"))
                    
            except Exception as e:
                logger.error(f"Error loading character: {e}")
                return redirect(url_for("gallery_page"))
        else:
            character = self.character_cache[character_id]
            
        return render_template(
            "character_view.html",
            title=f"Character: {character.name or 'Unnamed'}",
            character=character,
            character_json=json.dumps(character.to_dict())
        )
    
    def gallery_page(self):
        """Render the gallery page with all characters."""
        # Get all characters
        character_dir = os.path.join(self.output_dir, "characters")
        Path(character_dir).mkdir(parents=True, exist_ok=True)
        
        characters = []
        
        # Load characters from cache first
        for char_id, character in self.character_cache.items():
            characters.append({
                "id": char_id,
                "name": character.name or "Unnamed",
                "creation_date": character.creation_date,
                "gender": character.gender,
                "age_category": character.age_category,
                "hair_color": character.hair_color,
                "eye_color": character.eye_color,
                "anime_style": character.anime_style,
                "generation": character.generation,
                "thumbnail": self._get_character_thumbnail(char_id)
            })
            
        # Load characters from files
        for file in Path(character_dir).glob("*.json"):
            try:
                with open(file, 'r') as f:
                    character_data = json.load(f)
                    
                char_id = character_data.get("id")
                
                # Skip if already in cache
                if char_id in self.character_cache:
                    continue
                    
                characters.append({
                    "id": char_id,
                    "name": character_data.get("name", "Unnamed"),
                    "creation_date": character_data.get("creation_date", ""),
                    "gender": character_data.get("gender", ""),
                    "age_category": character_data.get("age_category", ""),
                    "hair_color": character_data.get("hair_color", ""),
                    "eye_color": character_data.get("eye_color", ""),
                    "anime_style": character_data.get("anime_style", ""),
                    "generation": character_data.get("generation", 0),
                    "thumbnail": self._get_character_thumbnail(char_id)
                })
                    
            except Exception as e:
                logger.error(f"Error loading character file {file}: {e}")
                
        # Sort by creation date (newest first)
        characters.sort(key=lambda x: x.get("creation_date", ""), reverse=True)
        
        return render_template(
            "gallery.html",
            title="Character Gallery",
            characters=characters
        )
    
    def settings_page(self):
        """Render the settings page."""
        return render_template(
            "settings.html",
            title="Settings",
            config=self.config
        )
    
    # API endpoints
    
    def api_create_character(self):
        """API endpoint to create a new character."""
        try:
            # Initialize components if needed
            if self.pipeline is None:
                self.initialize_components()
                
            # Get character attributes from request
            data = request.json
            
            # Create character
            if not data:
                # Create random character if no data provided
                character = CharacterAttributes.random(self.config_path)
            else:
                # Create from provided attributes
                character = CharacterAttributes()
                
                # Fill in attributes
                for key, value in data.items():
                    if hasattr(character, key):
                        setattr(character, key, value)
            
            # Generate character image
            images, output = self.image_generator.generate_images(
                attributes=character,
                num_samples=4  # Generate multiple samples
            )
            
            # Save images
            image_paths = self.image_generator.save_images(
                images=images,
                character=character,
                output=output,
                prefix="creation"
            )
            
            # Add to cache
            self.character_cache[character.id] = character
            
            # Add to active characters
            self.active_characters.append({
                "id": character.id,
                "name": character.name or "Unnamed",
                "image_path": image_paths[0] if image_paths else None
            })
            
            # Prepare response
            response = {
                "success": True,
                "character": character.to_dict(),
                "image_paths": image_paths
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error creating character: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    def api_random_character(self):
        """API endpoint to create a random character."""
        try:
            # Initialize components if needed
            if self.pipeline is None:
                self.initialize_components()
                
            # Create random character
            character = CharacterAttributes.random(self.config_path)
            
            # Generate character image
            images, output = self.image_generator.generate_images(
                attributes=character,
                num_samples=4  # Generate multiple samples
            )
            
            # Save images
            image_paths = self.image_generator.save_images(
                images=images,
                character=character,
                output=output,
                prefix="random"
            )
            
            # Add to cache
            self.character_cache[character.id] = character
            
            # Add to active characters
            self.active_characters.append({
                "id": character.id,
                "name": character.name or "Unnamed",
                "image_path": image_paths[0] if image_paths else None
            })
            
            # Prepare response
            response = {
                "success": True,
                "character": character.to_dict(),
                "image_paths": image_paths
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error creating random character: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    def api_evolve_character(self):
        """API endpoint to evolve a character."""
        try:
            # Initialize components if needed
            if self.evolution_engine is None:
                self.initialize_components()
                
            # Get request data
            data = request.json
            character_id = data.get("character_id")
            evolution_type = data.get("type", "single")
            
            # Check if character exists
            if character_id not in self.character_cache:
                return jsonify({
                    "success": False,
                    "error": "Character not found"
                }), 404
                
            character = self.character_cache[character_id]
            
            # Handle different evolution types
            if evolution_type == "single":
                # Evolve a single character
                evolved = self.evolution_engine.evolve_single(character)
                evolved_characters = [evolved]
                
            elif evolution_type == "pair":
                # Evolve with another character
                other_id = data.get("other_character_id")
                
                if other_id not in self.character_cache:
                    return jsonify({
                        "success": False,
                        "error": "Second character not found"
                    }), 404
                    
                other_character = self.character_cache[other_id]
                evolved = self.evolution_engine.evolve_pair(character, other_character)
                evolved_characters = [evolved]
                
            elif evolution_type == "generation":
                # Evolve a whole generation
                # Create a small population with the character and some variations
                population = [character]
                
                # Add a few mutations
                for _ in range(3):
                    population.append(character.mutate(0.3))
                    
                # Evolve the population
                evolved_characters = self.evolution_engine.evolve_generation(population)
                
            else:
                return jsonify({
                    "success": False,
                    "error": f"Unknown evolution type: {evolution_type}"
                }), 400
            
            # Generate images for evolved characters
            results = []
            
            for evolved_char in evolved_characters:
                # Generate character image
                images, output = self.image_generator.generate_images(
                    attributes=evolved_char,
                    num_samples=1
                )
                
                # Save images
                image_paths = self.image_generator.save_images(
                    images=images,
                    character=evolved_char,
                    output=output,
                    prefix="evolved"
                )
                
                # Add to cache
                self.character_cache[evolved_char.id] = evolved_char
                
                # Record evolution
                self.evolution_history.append({
                    "parent_id": character_id,
                    "child_id": evolved_char.id,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Add to results
                results.append({
                    "character": evolved_char.to_dict(),
                    "image_paths": image_paths
                })
            
            # Prepare response
            response = {
                "success": True,
                "results": results
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error evolving character: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    def api_regenerate_character(self):
        """API endpoint to regenerate character images."""
        try:
            # Initialize components if needed
            if self.image_generator is None:
                self.initialize_components()
                
            # Get request data
            data = request.json
            character_id = data.get("character_id")
            
            # Check if character exists
            if character_id not in self.character_cache:
                return jsonify({
                    "success": False,
                    "error": "Character not found"
                }), 404
                
            character = self.character_cache[character_id]
            
            # Get optional parameters
            num_samples = data.get("num_samples", 1)
            custom_prompt = data.get("custom_prompt")
            apply_style = data.get("style")
            
            # Generate character image
            images, output = self.image_generator.generate_images(
                attributes=character,
                num_samples=num_samples,
                custom_prompt=custom_prompt,
                apply_style=apply_style
            )
            
            # Save images
            image_paths = self.image_generator.save_images(
                images=images,
                character=character,
                output=output,
                prefix="regenerated"
            )
            
            # Prepare response
            response = {
                "success": True,
                "character": character.to_dict(),
                "image_paths": image_paths
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error regenerating character: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    def api_save_character(self):
        """API endpoint to save a character to disk."""
        try:
            # Get request data
            data = request.json
            character_id = data.get("character_id")
            
            # Check if character exists
            if character_id not in self.character_cache:
                return jsonify({
                    "success": False,
                    "error": "Character not found"
                }), 404
                
            character = self.character_cache[character_id]
            
            # Save character to file
            character_dir = os.path.join(self.output_dir, "characters")
            Path(character_dir).mkdir(parents=True, exist_ok=True)
            
            filename = f"{character_id}.json"
            filepath = os.path.join(character_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(character.to_dict(), f, indent=2)
                
            # Prepare response
            response = {
                "success": True,
                "filepath": filepath
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error saving character: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    def api_get_character(self, character_id):
        """API endpoint to get character data."""
        try:
            # Check if character exists in cache
            if character_id in self.character_cache:
                character = self.character_cache[character_id]
                
                # Find images for character
                image_paths = self._find_character_images(character_id)
                
                # Prepare response
                response = {
                    "success": True,
                    "character": character.to_dict(),
                    "image_paths": image_paths
                }
                
                return jsonify(response)
                
            # Try to load from file
            character_dir = os.path.join(self.output_dir, "characters")
            character_file = os.path.join(character_dir, f"{character_id}.json")
            
            if os.path.exists(character_file):
                with open(character_file, 'r') as f:
                    character_data = json.load(f)
                    
                # Create CharacterAttributes object
                character = CharacterAttributes.from_dict(character_data)
                
                # Add to cache
                self.character_cache[character_id] = character
                
                # Find images for character
                image_paths = self._find_character_images(character_id)
                
                # Prepare response
                response = {
                    "success": True,
                    "character": character.to_dict(),
                    "image_paths": image_paths
                }
                
                return jsonify(response)
            
            return jsonify({
                "success": False,
                "error": "Character not found"
            }), 404
            
        except Exception as e:
            logger.error(f"Error getting character: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    def api_get_styles(self):
        """API endpoint to get available anime styles."""
        try:
            # Initialize components if needed
            if self.style_library is None:
                self.initialize_components()
                
            # Get styles
            anime_styles = self.style_library.list_anime_styles()
            art_styles = self.style_library.list_art_styles()
            
            # Get detailed style info
            anime_style_info = {}
            for style in anime_styles:
                anime_style_info[style] = self.style_library.get_style_info(style)
                
            art_style_info = {}
            for style in art_styles:
                art_style_info[style] = self.style_library.get_style_info(style)
                
            # Prepare response
            response = {
                "success": True,
                "anime_styles": anime_styles,
                "art_styles": art_styles,
                "anime_style_info": anime_style_info,
                "art_style_info": art_style_info
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error getting styles: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    def api_get_image(self, image_id):
        """API endpoint to get image data."""
        try:
            # Find image file
            image_path = self._find_image_by_id(image_id)
            
            if not image_path:
                return jsonify({
                    "success": False,
                    "error": "Image not found"
                }), 404
                
            # Convert image to base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
                
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Determine image type
            image_type = os.path.splitext(image_path)[1].lower()
            if image_type == '.png':
                mime_type = 'image/png'
            elif image_type in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            else:
                mime_type = 'application/octet-stream'
                
            # Prepare response
            response = {
                "success": True,
                "image_data": f"data:{mime_type};base64,{base64_data}"
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error getting image: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    def api_get_lineage(self, character_id):
        """API endpoint to get character lineage."""
        try:
            # Initialize components if needed
            if self.evolution_engine is None:
                self.initialize_components()
                
            # Get lineage
            lineage = self.evolution_engine.get_lineage(character_id)
            
            # Prepare response
            response = {
                "success": True,
                "lineage": lineage
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error getting lineage: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Helper methods
    
    def _find_character_images(self, character_id):
        """Find all images for a character."""
        image_dir = os.path.join(self.output_dir, "images")
        
        # Check for character-specific directory
        char_dirs = list(Path(image_dir).glob(f"*_{character_id[:8]}"))
        
        if char_dirs:
            # Use first matching directory
            char_dir = char_dirs[0]
            image_files = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))
            return [str(path) for path in image_files]
            
        # Search in all subdirectories
        all_images = []
        for ext in ["png", "jpg"]:
            all_images.extend(list(Path(image_dir).glob(f"**/*_{character_id[:8]}*.{ext}")))
            
        return [str(path) for path in all_images]
    
    def _find_image_by_id(self, image_id):
        """Find an image by ID (partial match)."""
        image_dir = os.path.join(self.output_dir, "images")
        
        # Search in all subdirectories
        for ext in ["png", "jpg"]:
            matches = list(Path(image_dir).glob(f"**/*{image_id}*.{ext}"))
            if matches:
                return str(matches[0])
                
        return None
    
    def _get_character_thumbnail(self, character_id):
        """Get a thumbnail image for a character."""
        # Find images for character
        image_paths = self._find_character_images(character_id)
        
        if not image_paths:
            return None
            
        # Use the first image as thumbnail
        return image_paths[0]
    
    # File serving methods
    
    def serve_upload(self, filename):
        """Serve uploaded files."""
        try:
            path = os.path.join(self.upload_dir, filename)
            return send_file(path)
        except Exception as e:
            logger.error(f"Error serving upload {filename}: {e}")
            abort(404)
    
    def serve_output(self, filename):
        """Serve output files."""
        try:
            path = os.path.join(self.output_dir, filename)
            return send_file(path)
        except Exception as e:
            logger.error(f"Error serving output {filename}: {e}")
            abort(404)


# Main entry point
if __name__ == "__main__":
    # Create and run the web application
    app = AnimeEvolutionWebApp()
    app.run()