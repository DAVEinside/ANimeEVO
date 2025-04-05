"""
Voice generator for anime characters.
Generates character voices based on their attributes.
"""

import os
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from datetime import datetime
import yaml
import json
import tempfile
import random

from core.attributes.character_attributes import CharacterAttributes

# Conditional imports for TTS systems
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

# Try to import TTS-specific libraries
try:
    # This is a placeholder for TTS model imports
    # In a real implementation, you would import the actual TTS libraries
    TTS_AVAILABLE = False
except ImportError:
    TTS_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class VoiceGenerator:
    """
    Generates voice lines for anime characters.
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        output_dir: str = None,
        device: str = None
    ):
        """
        Initialize the voice generator.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for saving outputs (overrides config)
            device: Device to use (cuda, cpu)
        """
        # Check if required libraries are available
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("Audio libraries not available. Voice generation will be limited.")
            
        if not TTS_AVAILABLE:
            logger.warning("TTS models not available. Using fallback voice generation.")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.config['paths']['output_dir']
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create voice subdirectory
        self.voice_dir = os.path.join(self.output_dir, "voices")
        Path(self.voice_dir).mkdir(exist_ok=True)
        
        # Determine device
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Voice settings
        self.voice_settings = self.config['output']['voice']
        
        # Initialize voice models lazily
        self.tts_model = None
        self.vocoder = None
        self._load_voice_models()
        
        # Voice presets
        self.voice_presets = self._load_voice_presets()
    
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
                'output': {
                    'voice': {
                        'enabled': True,
                        'sample_rate': 22050,
                        'formats': ['wav', 'mp3'],
                    }
                },
                'paths': {
                    'output_dir': './outputs',
                },
                'multimodal': {
                    'voice': {
                        'model': 'tacotron2',
                        'vocoder': 'waveglow',
                        'language': 'japanese',
                        'voice_presets': [
                            'female_teen',
                            'female_adult',
                            'male_teen',
                            'male_adult',
                            'child'
                        ]
                    }
                }
            }
    
    def _load_voice_models(self):
        """Load TTS models if available."""
        if not TTS_AVAILABLE or not self.voice_settings['enabled']:
            return
            
        try:
            # Get voice model settings
            voice_config = self.config['multimodal']['voice']
            model_type = voice_config['model']
            vocoder_type = voice_config['vocoder']
            
            logger.info(f"Loading TTS model: {model_type} with vocoder: {vocoder_type}")
            
            # This is a placeholder for TTS model initialization
            # In a real implementation, you would load the actual models
            
            # For now, we'll set placeholders to indicate the models
            # would be loaded in a real implementation
            self.tts_model = model_type
            self.vocoder = vocoder_type
            
            logger.info("TTS models loaded")
                
        except Exception as e:
            logger.error(f"Error loading TTS models: {e}")
            logger.warning("Voice generation will be limited")
            
    def _load_voice_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load voice presets from config."""
        presets = {}
        
        # Load from config
        voice_config = self.config['multimodal'].get('voice', {})
        preset_names = voice_config.get('voice_presets', [])
        
        # Create basic presets
        for preset_name in preset_names:
            if preset_name == 'female_teen':
                presets[preset_name] = {
                    'gender': 'female',
                    'age': 'teen',
                    'pitch_shift': 0.3,
                    'speed': 1.05,
                    'energy': 1.1,
                    'breathiness': 0.2,
                    'description': 'Teenage female voice with higher pitch'
                }
            elif preset_name == 'female_adult':
                presets[preset_name] = {
                    'gender': 'female',
                    'age': 'adult',
                    'pitch_shift': 0.0,
                    'speed': 1.0,
                    'energy': 1.0,
                    'breathiness': 0.1,
                    'description': 'Adult female voice with natural tone'
                }
            elif preset_name == 'male_teen':
                presets[preset_name] = {
                    'gender': 'male',
                    'age': 'teen',
                    'pitch_shift': -0.1,
                    'speed': 1.05,
                    'energy': 1.2,
                    'breathiness': 0.1,
                    'description': 'Teenage male voice with occasional voice cracks'
                }
            elif preset_name == 'male_adult':
                presets[preset_name] = {
                    'gender': 'male',
                    'age': 'adult',
                    'pitch_shift': -0.3,
                    'speed': 0.95,
                    'energy': 1.0,
                    'breathiness': 0.05,
                    'description': 'Deep adult male voice'
                }
            elif preset_name == 'child':
                presets[preset_name] = {
                    'gender': 'neutral',
                    'age': 'child',
                    'pitch_shift': 0.5,
                    'speed': 1.1,
                    'energy': 1.2,
                    'breathiness': 0.3,
                    'description': 'High-pitched child voice with lots of energy'
                }
                
        # Try to load additional presets from a separate file
        preset_path = os.path.join(os.path.dirname(self.config_path), "voice_presets.json")
        if os.path.exists(preset_path):
            try:
                with open(preset_path, 'r') as f:
                    additional_presets = json.load(f)
                presets.update(additional_presets)
            except Exception as e:
                logger.error(f"Error loading additional voice presets: {e}")
                
        return presets
    
    def generate_voice(
        self,
        character: CharacterAttributes,
        text: str,
        preset: str = None,
        output_format: str = "wav",
        output_path: str = None,
        **kwargs
    ) -> str:
        """
        Generate a voice line for a character.
        
        Args:
            character: Character attributes
            text: Text to synthesize
            preset: Voice preset to use (if None, will be selected based on character)
            output_format: Output format (wav, mp3)
            output_path: Custom output path (optional)
            **kwargs: Additional parameters
            
        Returns:
            Path to saved voice file
        """
        # Set format from config if not in supported formats
        if output_format not in self.voice_settings['formats']:
            output_format = self.voice_settings['formats'][0]
            
        logger.info(f"Generating voice for character {character.id}: '{text}'")
        
        # Select preset if not provided
        if preset is None:
            preset = self._select_preset_for_character(character)
            
        # Check if preset exists
        if preset not in self.voice_presets:
            logger.warning(f"Preset '{preset}' not found. Using fallback.")
            preset = next(iter(self.voice_presets))
            
        # Use TTS models if available
        if TTS_AVAILABLE and self.tts_model is not None and self.vocoder is not None:
            return self._generate_tts_voice(
                text=text,
                preset=preset,
                output_format=output_format,
                output_path=output_path,
                **kwargs
            )
        else:
            # Fall back to basic voice generation
            return self._generate_fallback_voice(
                character=character,
                text=text,
                preset=preset,
                output_format=output_format,
                output_path=output_path
            )
    
    def _select_preset_for_character(self, character: CharacterAttributes) -> str:
        """Select the most appropriate voice preset based on character attributes."""
        # Get gender and age
        gender = character.gender.lower() if character.gender else "female"
        age = character.age_category.lower() if character.age_category else "teen"
        
        # Find matching preset
        if gender == "female" and age in ["teen", "young adult"]:
            return "female_teen"
        elif gender == "female" and age in ["adult", "elderly"]:
            return "female_adult"
        elif gender == "male" and age in ["teen", "young adult"]:
            return "male_teen"
        elif gender == "male" and age in ["adult", "elderly"]:
            return "male_adult"
        elif age == "child":
            return "child"
        else:
            # Default to female teen as common anime character voice
            return "female_teen"
    
    def _generate_tts_voice(
        self,
        text: str,
        preset: str,
        output_format: str = "wav",
        output_path: str = None,
        **kwargs
    ) -> str:
        """
        Generate voice using TTS models.
        
        Args:
            text: Text to synthesize
            preset: Voice preset to use
            output_format: Output format
            output_path: Custom output path
            **kwargs: Additional parameters
            
        Returns:
            Path to saved voice file
        """
        # This is a placeholder for actual TTS implementation
        # In a real implementation, you would use the loaded TTS models
        
        logger.warning("TTS voice generation not fully implemented. Using fallback.")
        
        # Create a placeholder audio file
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_{preset}_{timestamp}.{output_format}"
            output_path = os.path.join(self.voice_dir, filename)
            
        # Generate a placeholder tone
        if AUDIO_LIBS_AVAILABLE:
            # Generate a simple tone
            sample_rate = self.voice_settings['sample_rate']
            duration = len(text) * 0.1  # Rough approximation of duration based on text length
            
            # Create a simple tone
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Generate different tones based on preset
            if "female" in preset:
                frequency = 220.0  # A3
            elif "male" in preset:
                frequency = 110.0  # A2
            elif "child" in preset:
                frequency = 440.0  # A4
            else:
                frequency = 220.0
                
            # Create a simple sine wave
            audio = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Apply some amplitude modulation to simulate speech rhythm
            mod_freq = 2.0  # Modulation frequency for rhythm
            audio = audio * (0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t / duration))
            
            # Save the audio
            sf.write(output_path, audio, sample_rate)
            
            logger.info(f"Saved placeholder voice to {output_path}")
            return output_path
        else:
            logger.error("Cannot generate voice - audio libraries not available")
            return None
    
    def _generate_fallback_voice(
        self,
        character: CharacterAttributes,
        text: str,
        preset: str,
        output_format: str = "wav",
        output_path: str = None
    ) -> str:
        """
        Generate a fallback voice when TTS models aren't available.
        Creates a simple audio file with tones representing speech.
        
        Args:
            character: Character attributes
            text: Text to synthesize
            preset: Voice preset to use
            output_format: Output format
            output_path: Custom output path
            
        Returns:
            Path to saved voice file
        """
        if not AUDIO_LIBS_AVAILABLE:
            logger.error("Cannot generate fallback voice - audio libraries not available")
            return None
            
        try:
            # Create output path if not provided
            if output_path is None:
                char_name = character.name.lower().replace(" ", "_") if character.name else "unnamed"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"voice_{char_name}_{character.id[:8]}_{timestamp}.{output_format}"
                output_path = os.path.join(self.voice_dir, filename)
                
            # Get preset parameters
            preset_params = self.voice_presets.get(preset, self.voice_presets[next(iter(self.voice_presets))])
            
            # Generate basic audio
            sample_rate = self.voice_settings['sample_rate']
            
            # Estimate duration based on text length and speed
            words = len(text.split())
            duration = words * 0.3 * (1.0 / preset_params.get('speed', 1.0))
            duration = max(1.0, min(60.0, duration))  # Limit between 1-60 seconds
            
            # Create time array
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Base frequency depends on gender/age
            if preset_params.get('gender') == 'female':
                base_freq = 200.0 + 20.0 * preset_params.get('pitch_shift', 0.0)
            elif preset_params.get('gender') == 'male':
                base_freq = 120.0 + 20.0 * preset_params.get('pitch_shift', 0.0)
            else:  # child or neutral
                base_freq = 300.0 + 30.0 * preset_params.get('pitch_shift', 0.0)
                
            # Create a more complex tone with harmonics
            audio = np.zeros_like(t)
            
            # Add fundamental frequency
            audio += 0.7 * np.sin(2 * np.pi * base_freq * t)
            
            # Add harmonics
            audio += 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)
            audio += 0.05 * np.sin(2 * np.pi * base_freq * 3 * t)
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Create speech-like rhythm by segmenting into syllables
            syllables = len(text) // 2  # Rough approximation
            syllable_duration = duration / syllables
            
            # Create amplitude envelope
            envelope = np.ones_like(t)
            
            for i in range(syllables):
                start_idx = int((i * syllable_duration) * sample_rate)
                end_idx = int(((i + 1) * syllable_duration) * sample_rate)
                
                # Apply a rise and fall pattern for each syllable
                if end_idx <= len(envelope):
                    syllable_t = np.linspace(0, 1, end_idx - start_idx)
                    # Rise quickly, fall slowly
                    syllable_env = 0.1 + 0.9 * np.sin(syllable_t * np.pi)
                    envelope[start_idx:end_idx] = syllable_env
                    
                    # Add small random pitch variations
                    pitch_var = 1.0 + 0.05 * np.sin(2 * np.pi * 4 * syllable_t)
                    audio[start_idx:end_idx] *= pitch_var
            
            # Apply envelope
            audio = audio * envelope
            
            # Apply energy parameter
            energy = preset_params.get('energy', 1.0)
            audio = audio * energy
            
            # Apply breathiness
            breathiness = preset_params.get('breathiness', 0.1)
            if breathiness > 0:
                noise = np.random.randn(len(audio)) * breathiness
                audio = (1.0 - breathiness) * audio + noise
                
            # Normalize again
            audio = 0.9 * audio / np.max(np.abs(audio))
            
            # Add a slight fade in/out
            fade_samples = int(0.1 * sample_rate)
            if len(audio) > 2 * fade_samples:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                audio[:fade_samples] *= fade_in
                audio[-fade_samples:] *= fade_out
            
            # Save the audio
            sf.write(output_path, audio, sample_rate)
            
            logger.info(f"Saved fallback voice to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating fallback voice: {e}")
            return None
    
    def generate_character_lines(
        self,
        character: CharacterAttributes,
        line_type: str = "introduction",
        output_format: str = "wav",
        output_dir: str = None
    ) -> List[str]:
        """
        Generate common character voice lines based on their attributes.
        
        Args:
            character: Character attributes
            line_type: Type of lines to generate
            output_format: Output format
            output_dir: Custom output directory
            
        Returns:
            List of paths to saved voice files
        """
        voice_lines = []
        
        # Select preset based on character
        preset = self._select_preset_for_character(character)
        
        # Generate appropriate text based on character and line type
        texts = self._generate_character_text(character, line_type)
        
        # Create custom output directory if provided
        if output_dir:
            dir_path = output_dir
        else:
            char_name = character.name.lower().replace(" ", "_") if character.name else "unnamed"
            dir_path = os.path.join(self.voice_dir, f"{char_name}_{character.id[:8]}")
            
        Path(dir_path).mkdir(exist_ok=True)
        
        # Generate voice for each text
        for i, text in enumerate(texts):
            filename = f"{line_type}_{i+1}.{output_format}"
            output_path = os.path.join(dir_path, filename)
            
            voice_path = self.generate_voice(
                character=character,
                text=text,
                preset=preset,
                output_format=output_format,
                output_path=output_path
            )
            
            if voice_path:
                voice_lines.append(voice_path)
                
        return voice_lines
    
    def _generate_character_text(self, character: CharacterAttributes, line_type: str) -> List[str]:
        """Generate appropriate text lines based on character attributes."""
        texts = []
        
        name = character.name if character.name else "Unnamed"
        gender = character.gender if character.gender else "female"
        personality = character.personality if character.personality else []
        
        # Convert personality to traits for text generation
        traits = []
        for trait in personality:
            if trait.lower() in ["cheerful", "energetic", "outgoing"]:
                traits.append("cheerful")
            elif trait.lower() in ["serious", "calm", "logical"]:
                traits.append("serious")
            elif trait.lower() in ["shy", "cautious", "quiet"]:
                traits.append("shy")
            else:
                traits.append(trait.lower())
        
        # Default to cheerful if no traits
        if not traits:
            traits = ["cheerful"]
            
        if line_type == "introduction":
            # Introduction lines
            if "cheerful" in traits:
                texts.append(f"Hi there! I'm {name}! Nice to meet you!")
                texts.append(f"Hello! The name's {name}. Let's be friends!")
            elif "serious" in traits:
                texts.append(f"I am {name}. Pleased to make your acquaintance.")
                texts.append(f"My name is {name}. I look forward to working with you.")
            elif "shy" in traits:
                texts.append(f"Um... I'm {name}... N-nice to meet you...")
                texts.append(f"H-hello... I'm {name}...")
            else:
                texts.append(f"I'm {name}. Nice to meet you.")
                texts.append(f"Hello, my name is {name}.")
                
        elif line_type == "greeting":
            # Greeting lines
            if "cheerful" in traits:
                texts.append("Good morning! It's a beautiful day!")
                texts.append("Hey there! How's it going?")
            elif "serious" in traits:
                texts.append("Good day. I trust you are well?")
                texts.append("Greetings. I hope your day is productive.")
            elif "shy" in traits:
                texts.append("G-good morning...")
                texts.append("H-hello again...")
            else:
                texts.append("Hello there.")
                texts.append("Good to see you.")
                
        elif line_type == "battle":
            # Battle lines for action characters
            if "brave" in traits or "determined" in traits:
                texts.append("I won't lose! I'll give it everything I've got!")
                texts.append("This is where I show my true power!")
            elif "cheerful" in traits:
                texts.append("Let's have a good fight!")
                texts.append("I'm not holding back! Here I come!")
            elif "serious" in traits:
                texts.append("I shall defeat you with precision.")
                texts.append("Your defeat is inevitable. Prepare yourself.")
            else:
                texts.append("I'll do my best in this battle!")
                texts.append("Let's fight with honor!")
                
        else:
            # Default/miscellaneous lines
            texts.append(f"This is {name} speaking.")
            texts.append(f"Hello from {name}.")
            
        return texts
    
    def mix_voice_with_background(
        self,
        voice_path: str,
        background_path: str,
        output_path: str = None,
        volume_ratio: float = 0.2
    ) -> str:
        """
        Mix a voice line with background music/effects.
        
        Args:
            voice_path: Path to voice file
            background_path: Path to background audio file
            output_path: Path to save mixed audio
            volume_ratio: Ratio of background to voice volume
            
        Returns:
            Path to saved mixed audio file
        """
        if not AUDIO_LIBS_AVAILABLE:
            logger.error("Cannot mix audio - audio libraries not available")
            return voice_path
            
        try:
            # Load the voice and background
            voice, voice_sr = librosa.load(voice_path, sr=None)
            background, bg_sr = librosa.load(background_path, sr=None)
            
            # Resample background to match voice sample rate if needed
            if bg_sr != voice_sr:
                background = librosa.resample(background, orig_sr=bg_sr, target_sr=voice_sr)
                
            # Adjust background length to match voice
            voice_length = len(voice)
            bg_length = len(background)
            
            if bg_length > voice_length:
                # Truncate background
                background = background[:voice_length]
            elif bg_length < voice_length:
                # Loop background if needed
                repeats = int(np.ceil(voice_length / bg_length))
                background = np.tile(background, repeats)[:voice_length]
                
            # Mix the audio with volume adjustment
            mixed = voice + background * volume_ratio
            
            # Normalize
            mixed = mixed / np.max(np.abs(mixed))
            
            # Create output path if not provided
            if output_path is None:
                voice_file = os.path.basename(voice_path)
                voice_name = os.path.splitext(voice_file)[0]
                output_path = os.path.join(
                    os.path.dirname(voice_path),
                    f"{voice_name}_with_bg.wav"
                )
                
            # Save the mixed audio
            sf.write(output_path, mixed, voice_sr)
            
            logger.info(f"Saved mixed audio to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error mixing audio: {e}")
            return voice_path