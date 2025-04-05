"""
Anime style definitions and utilities for character generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
import yaml
import os
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class AnimeStyle:
    """Definition of an anime visual style."""
    name: str
    description: str
    era: str = ""
    target_audience: str = ""
    common_themes: List[str] = field(default_factory=list)
    color_palette: List[str] = field(default_factory=list)
    character_traits: List[str] = field(default_factory=list)
    visual_characteristics: List[str] = field(default_factory=list)
    reference_anime: List[str] = field(default_factory=list)
    prompt_modifiers: List[str] = field(default_factory=list)
    negative_prompt_modifiers: List[str] = field(default_factory=list)

@dataclass
class ArtStyle:
    """Definition of an art style within anime."""
    name: str
    description: str
    visual_characteristics: List[str] = field(default_factory=list)
    artists: List[str] = field(default_factory=list)
    prompt_modifiers: List[str] = field(default_factory=list)
    negative_prompt_modifiers: List[str] = field(default_factory=list)


class AnimeStyleLibrary:
    """Library of anime styles and art styles for reference and generation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the anime style library.
        
        Args:
            config_path: Path to the configuration file
        """
        self.anime_styles: Dict[str, AnimeStyle] = {}
        self.art_styles: Dict[str, ArtStyle] = {}
        
        # Load from configuration
        self._load_from_config(config_path)
        
        # Initialize built-in styles if none loaded
        if not self.anime_styles:
            self._initialize_default_styles()
    
    def _load_from_config(self, config_path: str):
        """Load styles from configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Check for style references in config
            style_references = config.get('style_transfer', {}).get('styles', [])
            
            for style in style_references:
                name = style.get('name', '')
                if name:
                    # Create a simple AnimeStyle for each style in config
                    self.anime_styles[name] = AnimeStyle(
                        name=name,
                        description=f"{name} anime style",
                        prompt_modifiers=[f"{name} style", f"{name} anime"],
                        negative_prompt_modifiers=[]
                    )
                    
            # Try to load more detailed style definitions if available
            style_config_path = os.path.join(os.path.dirname(config_path), "styles.yaml")
            if os.path.exists(style_config_path):
                self._load_from_style_config(style_config_path)
                
        except Exception as e:
            logger.error(f"Error loading styles from config: {e}")
    
    def _load_from_style_config(self, style_config_path: str):
        """Load detailed style definitions from a style-specific config file."""
        try:
            with open(style_config_path, 'r') as f:
                styles_config = yaml.safe_load(f)
                
            # Load anime styles
            anime_styles = styles_config.get('anime_styles', [])
            for style_data in anime_styles:
                name = style_data.get('name')
                if name:
                    self.anime_styles[name] = AnimeStyle(
                        name=name,
                        description=style_data.get('description', f"{name} anime style"),
                        era=style_data.get('era', ''),
                        target_audience=style_data.get('target_audience', ''),
                        common_themes=style_data.get('common_themes', []),
                        color_palette=style_data.get('color_palette', []),
                        character_traits=style_data.get('character_traits', []),
                        visual_characteristics=style_data.get('visual_characteristics', []),
                        reference_anime=style_data.get('reference_anime', []),
                        prompt_modifiers=style_data.get('prompt_modifiers', [f"{name} style"]),
                        negative_prompt_modifiers=style_data.get('negative_prompt_modifiers', [])
                    )
                    
            # Load art styles
            art_styles = styles_config.get('art_styles', [])
            for style_data in art_styles:
                name = style_data.get('name')
                if name:
                    self.art_styles[name] = ArtStyle(
                        name=name,
                        description=style_data.get('description', f"{name} art style"),
                        visual_characteristics=style_data.get('visual_characteristics', []),
                        artists=style_data.get('artists', []),
                        prompt_modifiers=style_data.get('prompt_modifiers', [f"{name} style"]),
                        negative_prompt_modifiers=style_data.get('negative_prompt_modifiers', [])
                    )
                    
            logger.info(f"Loaded {len(self.anime_styles)} anime styles and {len(self.art_styles)} art styles from {style_config_path}")
                    
        except Exception as e:
            logger.error(f"Error loading from style config: {e}")
    
    def _initialize_default_styles(self):
        """Initialize built-in default styles."""
        # Shonen style
        self.anime_styles["shonen"] = AnimeStyle(
            name="shonen",
            description="Action-oriented anime style targeting young male audience",
            era="all eras",
            target_audience="young males",
            common_themes=["action", "adventure", "friendship", "perseverance", "training"],
            color_palette=["vibrant", "high contrast", "bold colors"],
            character_traits=["determined", "strong-willed", "enthusiastic", "heroic"],
            visual_characteristics=[
                "dynamic action scenes", 
                "exaggerated expressions", 
                "detailed power effects",
                "emphasis on action and movement"
            ],
            reference_anime=["Dragon Ball", "Naruto", "One Piece", "My Hero Academia"],
            prompt_modifiers=["shonen style", "action anime", "vibrant colors", "dynamic pose"],
            negative_prompt_modifiers=["muted colors", "static pose", "photorealistic"]
        )
        
        # Shojo style
        self.anime_styles["shojo"] = AnimeStyle(
            name="shojo",
            description="Emotion-focused anime style targeting young female audience",
            era="all eras",
            target_audience="young females",
            common_themes=["romance", "friendship", "emotions", "personal growth", "relationships"],
            color_palette=["pastel colors", "soft lighting", "delicate shading"],
            character_traits=["emotional", "sensitive", "kind", "determined"],
            visual_characteristics=[
                "flowery backgrounds", 
                "sparkling effects", 
                "large expressive eyes",
                "delicate character designs",
                "emphasis on beauty and emotions"
            ],
            reference_anime=["Sailor Moon", "Fruits Basket", "Ouran High School Host Club", "Kimi ni Todoke"],
            prompt_modifiers=["shojo style", "romantic anime", "pastel colors", "flowery background", "sparkles"],
            negative_prompt_modifiers=["dark atmosphere", "gritty", "realistic"]
        )
        
        # Seinen style
        self.anime_styles["seinen"] = AnimeStyle(
            name="seinen",
            description="Mature anime style targeting adult male audience",
            era="all eras",
            target_audience="adult males",
            common_themes=["psychological", "mature themes", "complex narratives", "social commentary"],
            color_palette=["realistic", "muted", "darker tones", "high contrast"],
            character_traits=["complex", "flawed", "mature", "introspective"],
            visual_characteristics=[
                "realistic proportions", 
                "detailed artwork", 
                "subtle expressions",
                "emphasis on atmosphere"
            ],
            reference_anime=["Berserk", "Ghost in the Shell", "Vinland Saga", "Monster"],
            prompt_modifiers=["seinen style", "mature anime", "realistic proportions", "detailed"],
            negative_prompt_modifiers=["childish", "exaggerated proportions", "overly cute"]
        )
        
        # Josei style
        self.anime_styles["josei"] = AnimeStyle(
            name="josei",
            description="Mature anime style targeting adult female audience",
            era="all eras",
            target_audience="adult females",
            common_themes=["slice of life", "relationships", "career", "realistic romance"],
            color_palette=["subtle", "sophisticated", "earthy tones"],
            character_traits=["independent", "mature", "realistic", "complex"],
            visual_characteristics=[
                "elegant character designs", 
                "realistic body proportions", 
                "detailed eyes with smaller irises",
                "emphasis on facial expressions"
            ],
            reference_anime=["Nana", "Paradise Kiss", "Usagi Drop", "Chihayafuru"],
            prompt_modifiers=["josei style", "elegant anime", "mature female protagonist", "sophisticated"],
            negative_prompt_modifiers=["childish", "overly cute", "unrealistic proportions"]
        )
        
        # Isekai style
        self.anime_styles["isekai"] = AnimeStyle(
            name="isekai",
            description="Fantasy world transportation anime style",
            era="2010s-present",
            target_audience="teens and young adults",
            common_themes=["fantasy", "adventure", "other world", "video game mechanics", "power fantasy"],
            color_palette=["fantasy colors", "vibrant", "saturated", "magical effects"],
            character_traits=["protagonist from another world", "adaptable", "special powers"],
            visual_characteristics=[
                "fantasy world setting", 
                "elaborate costumes", 
                "RPG-like elements",
                "magical effects and creatures"
            ],
            reference_anime=["Sword Art Online", "Re:Zero", "Konosuba", "That Time I Got Reincarnated as a Slime"],
            prompt_modifiers=["isekai style", "fantasy anime", "magical world", "adventurer"],
            negative_prompt_modifiers=["modern setting", "realistic", "mundane"]
        )
        
        # Mecha style
        self.anime_styles["mecha"] = AnimeStyle(
            name="mecha",
            description="Robot and mechanical themed anime style",
            era="1980s-present",
            target_audience="male teens and young adults",
            common_themes=["giant robots", "war", "technology", "human connection to machines"],
            color_palette=["metallic", "industrial", "high tech", "neon highlights"],
            character_traits=["pilots", "soldiers", "engineers", "determined"],
            visual_characteristics=[
                "detailed mechanical designs", 
                "giant robots", 
                "technological interfaces",
                "dynamic battle scenes",
                "contrast between human and mechanical elements"
            ],
            reference_anime=["Gundam series", "Evangelion", "Code Geass", "Gurren Lagann"],
            prompt_modifiers=["mecha style", "robot anime", "mechanical details", "technical design"],
            negative_prompt_modifiers=["organic", "natural", "simplistic"]
        )
        
        # Chibi style
        self.anime_styles["chibi"] = AnimeStyle(
            name="chibi",
            description="Super-deformed cute miniature anime style",
            era="all eras",
            target_audience="all ages",
            common_themes=["comedy", "cute moments", "simplified stories"],
            color_palette=["bright", "cute", "pastel", "playful colors"],
            character_traits=["cute", "simplified", "exaggerated emotions"],
            visual_characteristics=[
                "super-deformed proportions", 
                "oversized heads", 
                "small bodies",
                "simplified features",
                "exaggerated expressions"
            ],
            reference_anime=["Lucky Star chibi scenes", "Himouto! Umaru-chan", "Nendoroid style"],
            prompt_modifiers=["chibi style", "super-deformed", "cute anime", "big head", "small body"],
            negative_prompt_modifiers=["realistic proportions", "detailed features", "mature"]
        )
        
        # 80s anime style
        self.anime_styles["80s_anime"] = AnimeStyle(
            name="80s_anime",
            description="Classic 1980s anime visual style",
            era="1980s",
            target_audience="varied",
            common_themes=["mecha", "sci-fi", "fantasy", "adventure"],
            color_palette=["vibrant", "high contrast", "strong shadows"],
            character_traits=["varied by genre"],
            visual_characteristics=[
                "hand-drawn appearance", 
                "detailed line work", 
                "film grain",
                "cel shading",
                "dramatic lighting",
                "painted backgrounds"
            ],
            reference_anime=["Akira", "Dragon Ball", "Macross", "Urusei Yatsura"],
            prompt_modifiers=["80s anime style", "retro anime", "vintage anime", "cel-shaded", "hand-drawn"],
            negative_prompt_modifiers=["digital art", "modern anime", "3D rendered"]
        )
        
        # 90s anime style
        self.anime_styles["90s_anime"] = AnimeStyle(
            name="90s_anime",
            description="Classic 1990s anime visual style",
            era="1990s",
            target_audience="varied",
            common_themes=["magical girl", "mecha", "cyberpunk", "psychological"],
            color_palette=["colorful", "rich colors", "distinctive shadows"],
            character_traits=["varied by genre"],
            visual_characteristics=[
                "distinctive character designs", 
                "detailed backgrounds", 
                "fluid animation style",
                "characteristic thick outlines"
            ],
            reference_anime=["Sailor Moon", "Evangelion", "Cowboy Bebop", "Ghost in the Shell"],
            prompt_modifiers=["90s anime style", "classic anime", "vintage 90s", "VHS anime look"],
            negative_prompt_modifiers=["digital art", "modern anime style", "3D rendered"]
        )
        
        # Modern anime style
        self.anime_styles["modern_anime"] = AnimeStyle(
            name="modern_anime",
            description="Contemporary anime visual style from 2010s onward",
            era="2010s-present",
            target_audience="varied",
            common_themes=["varied by genre"],
            color_palette=["digital colors", "clean gradients", "bright palette"],
            character_traits=["varied by genre"],
            visual_characteristics=[
                "clean lines", 
                "digital coloring", 
                "smooth gradients",
                "detailed hair rendering",
                "elaborate effects"
            ],
            reference_anime=["Demon Slayer", "Attack on Titan", "Your Name", "Jujutsu Kaisen"],
            prompt_modifiers=["modern anime style", "contemporary anime", "digital anime", "clean lines"],
            negative_prompt_modifiers=["retro anime", "hand-drawn", "cel-shaded"]
        )
        
        # Art styles
        
        # Watercolor anime style
        self.art_styles["watercolor"] = ArtStyle(
            name="watercolor",
            description="Watercolor painting inspired anime style",
            visual_characteristics=[
                "soft color bleeding", 
                "transparent layers", 
                "visible brush textures",
                "organic color blending"
            ],
            artists=["Makoto Shinkai backgrounds", "Violet Evergarden backgrounds"],
            prompt_modifiers=["watercolor anime", "watercolor painting style", "soft watercolor rendering"],
            negative_prompt_modifiers=["sharp lines", "cell shaded", "digital art"]
        )
        
        # Sketch style
        self.art_styles["sketch"] = ArtStyle(
            name="sketch",
            description="Sketch-like anime drawing style",
            visual_characteristics=[
                "visible pencil/pen lines", 
                "minimal coloring", 
                "rough linework",
                "unfinished appearance"
            ],
            artists=["animation key frames", "manga drafts"],
            prompt_modifiers=["anime sketch", "rough drawing", "pencil sketch anime", "line art"],
            negative_prompt_modifiers=["detailed coloring", "painted", "polished"]
        )
        
        # Minimalist style
        self.art_styles["minimalist"] = ArtStyle(
            name="minimalist",
            description="Clean, simplified anime style with minimal details",
            visual_characteristics=[
                "clean lines", 
                "simplified features", 
                "minimal shading",
                "reduced color palette",
                "essential details only"
            ],
            artists=["Mob Psycho 100 style", "ONE's style"],
            prompt_modifiers=["minimalist anime", "simplified style", "clean design", "essential details"],
            negative_prompt_modifiers=["overly detailed", "complex shading", "busy artwork"]
        )
        
        logger.info(f"Initialized {len(self.anime_styles)} default anime styles and {len(self.art_styles)} art styles")
    
    def get_anime_style(self, name: str) -> Optional[AnimeStyle]:
        """Get an anime style by name."""
        return self.anime_styles.get(name.lower())
    
    def get_art_style(self, name: str) -> Optional[ArtStyle]:
        """Get an art style by name."""
        return self.art_styles.get(name.lower())
    
    def get_style_prompt_modifiers(self, anime_style: str = None, art_style: str = None) -> List[str]:
        """
        Get prompt modifiers for the given styles.
        
        Args:
            anime_style: Anime style name
            art_style: Art style name
            
        Returns:
            List of prompt modifier strings
        """
        modifiers = []
        
        # Add anime style modifiers
        if anime_style and anime_style in self.anime_styles:
            modifiers.extend(self.anime_styles[anime_style].prompt_modifiers)
            
        # Add art style modifiers
        if art_style and art_style in self.art_styles:
            modifiers.extend(self.art_styles[art_style].prompt_modifiers)
            
        return modifiers
    
    def get_style_negative_prompt_modifiers(self, anime_style: str = None, art_style: str = None) -> List[str]:
        """
        Get negative prompt modifiers for the given styles.
        
        Args:
            anime_style: Anime style name
            art_style: Art style name
            
        Returns:
            List of negative prompt modifier strings
        """
        modifiers = []
        
        # Add anime style negative modifiers
        if anime_style and anime_style in self.anime_styles:
            modifiers.extend(self.anime_styles[anime_style].negative_prompt_modifiers)
            
        # Add art style negative modifiers
        if art_style and art_style in self.art_styles:
            modifiers.extend(self.art_styles[art_style].negative_prompt_modifiers)
            
        return modifiers
    
    def list_anime_styles(self) -> List[str]:
        """List all available anime style names."""
        return list(self.anime_styles.keys())
    
    def list_art_styles(self) -> List[str]:
        """List all available art style names."""
        return list(self.art_styles.keys())
    
    def get_style_info(self, style_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a style (anime or art).
        
        Args:
            style_name: Name of the style
            
        Returns:
            Dictionary with style information
        """
        # Check anime styles
        anime_style = self.get_anime_style(style_name)
        if anime_style:
            return {
                "type": "anime_style",
                "name": anime_style.name,
                "description": anime_style.description,
                "era": anime_style.era,
                "target_audience": anime_style.target_audience,
                "common_themes": anime_style.common_themes,
                "visual_characteristics": anime_style.visual_characteristics,
                "reference_anime": anime_style.reference_anime
            }
            
        # Check art styles
        art_style = self.get_art_style(style_name)
        if art_style:
            return {
                "type": "art_style",
                "name": art_style.name,
                "description": art_style.description,
                "visual_characteristics": art_style.visual_characteristics,
                "artists": art_style.artists
            }
            
        # Style not found
        return {"error": f"Style '{style_name}' not found"}