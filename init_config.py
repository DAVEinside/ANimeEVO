"""
Initialization script to ensure configuration files are properly set up.
"""

import os
import yaml
from pathlib import Path

def ensure_config_files():
    """
    Ensure that the configuration files and directories exist.
    """
    print("Checking and setting up configuration files...")
    
    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Main config file
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        print(f"Creating {config_file}...")
        
        # Default config
        config = {
            'app': {
                'name': "Anime Character Evolution System",
                'version': "0.1.0",
                'debug': True,
                'log_level': "INFO"
            },
            'paths': {
                'models_dir': "./models",
                'output_dir': "./outputs",
                'cache_dir': "./cache",
                'data_dir': "./data",
                'temp_dir': "./temp"
            },
            'interface': {
                'web': {
                    'enabled': True,
                    'host': "0.0.0.0",
                    'port': 8080,
                    'template_dir': "./interface/templates",
                    'static_dir': "./interface/static"
                },
                'cli': {
                    'enabled': True
                }
            },
            'diffusion': {
                'base_model': "runwayml/stable-diffusion-v1-5",
                'anime_model': "runwayml/stable-diffusion-v1-5",
                'inference': {
                    'steps': 30,
                    'guidance_scale': 7.5,
                    'width': 512,
                    'height': 512,
                    'sampler': 'DDIM',
                    'clip_skip': 1,
                    'seed': -1
                }
            },
            'output': {
                'image': {
                    'formats': ["png", "jpg"],
                    'resolution': [512, 512],
                    'high_res_multiple': 2,
                    'samples_per_character': 4
                },
                'animation': {
                    'enabled': True,
                    'fps': 24,
                    'duration': 3.0,
                    'formats': ["mp4", "gif"]
                },
                'voice': {
                    'enabled': True,
                    'sample_rate': 22050,
                    'formats': ["wav", "mp3"]
                },
                'model_3d': {
                    'enabled': False
                }
            },
            'character': {
                'attributes': {
                    'physical': [
                        'hair_color', 'eye_color', 'skin_tone', 'body_type',
                        'height', 'age', 'gender', 'distinctive_features'
                    ],
                    'personality': [
                        'archetype', 'traits', 'temperament', 'motivation', 'background'
                    ],
                    'style': [
                        'anime_genre', 'art_style', 'era', 'color_palette'
                    ]
                }
            },
            'evolution': {
                'population_size': 8,
                'generations': 5,
                'mutation_rate': 0.15,
                'crossover_rate': 0.75,
                'elite_size': 2,
                'selection_method': "tournament",
                'tournament_size': 3,
                'fitness_weights': {
                    'user_feedback': 0.7,
                    'style_consistency': 0.2,
                    'diversity': 0.1
                }
            },
            'resources': {
                'gpu_id': 0,
                'cpu_threads': 4,
                'memory_limit': "8GB",
                'batch_size': 4,
                'precision': "fp16"
            },
            'logging': {
                'save_generations': True,
                'save_evolution_history': True
            },
            'style_transfer': {
                'enabled': True,
                'styles': [
                    {'name': 'shonen', 'reference': 'shonen_style.png', 'strength': 0.8},
                    {'name': 'shojo', 'reference': 'shojo_style.png', 'strength': 0.8},
                    {'name': 'seinen', 'reference': 'seinen_style.png', 'strength': 0.8},
                    {'name': 'chibi', 'reference': 'chibi_style.png', 'strength': 0.9},
                    {'name': '90s_anime', 'reference': '90s_anime.png', 'strength': 0.7},
                    {'name': 'modern_anime', 'reference': 'modern_anime.png', 'strength': 0.7}
                ]
            },
            'multimodal': {
                'model_3d': {
                    'type': "img2mesh",
                    'resolution': 256,
                    'view_angles': 8,
                    'texture_resolution': 1024
                },
                'animation': {
                    'model': "animatediff",
                    'motion_strength': 0.6,
                    'motion_modules': ["mm_sd_v15"]
                },
                'voice': {
                    'model': "tacotron2",
                    'vocoder': "waveglow",
                    'language': "japanese",
                    'voice_presets': [
                        "female_teen", "female_adult",
                        "male_teen", "male_adult", "child"
                    ]
                }
            },
            'attribute_conditioning': {
                'use_textual_inversion': True,
                'use_dreambooth_concepts': True,
                'embedding_dir': "./embeddings",
                'concept_dir': "./concepts"
            }
        }
        
        # Write config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        print(f"Config file {config_file} already exists.")
    
    # Model config file
    model_config_file = config_dir / "model_config.yaml"
    if not model_config_file.exists():
        print(f"Creating {model_config_file}...")
        
        # Default model config
        model_config = {
            'diffusion': {
                'base_model': "runwayml/stable-diffusion-v1-5",
                'anime_model': "animefull-latest",
                'anime_lora': "anime_style_lora",
                'inference': {
                    'steps': 30,
                    'guidance_scale': 7.5,
                    'width': 512,
                    'height': 512,
                    'batch_size': 4,
                    'sampler': "DPM++ 2M Karras",
                    'clip_skip': 2,
                    'eta': 0.0,
                    'seed': -1
                },
                'training': {
                    'learning_rate': 1.0e-5,
                    'train_batch_size': 4,
                    'max_train_steps': 10000,
                    'save_interval': 500
                }
            },
            'multimodal': {
                'model_3d': {
                    'type': "img2mesh",
                    'resolution': 256,
                    'view_angles': 8,
                    'texture_resolution': 1024
                },
                'animation': {
                    'model': "animatediff",
                    'motion_strength': 0.6,
                    'motion_modules': ["mm_sd_v15"]
                },
                'voice': {
                    'model': "tacotron2",
                    'vocoder': "waveglow",
                    'language': "japanese",
                    'voice_presets': [
                        "female_teen", "female_adult",
                        "male_teen", "male_adult", "child"
                    ]
                }
            },
            'style_transfer': {
                'enabled': True,
                'method': "controlnet",
                'style_reference_dir': "./data/style_references",
                'styles': [
                    {'name': 'shonen', 'reference': 'shonen_style.png', 'strength': 0.8},
                    {'name': 'shojo', 'reference': 'shojo_style.png', 'strength': 0.8},
                    {'name': 'seinen', 'reference': 'seinen_style.png', 'strength': 0.8},
                    {'name': 'isekai', 'reference': 'isekai_style.png', 'strength': 0.75},
                    {'name': 'mecha', 'reference': 'mecha_style.png', 'strength': 0.85},
                    {'name': 'chibi', 'reference': 'chibi_style.png', 'strength': 0.9},
                    {'name': '90s_anime', 'reference': '90s_anime.png', 'strength': 0.7},
                    {'name': 'modern_anime', 'reference': 'modern_anime.png', 'strength': 0.7}
                ]
            },
            'attribute_conditioning': {
                'use_textual_inversion': True,
                'use_dreambooth_concepts': True,
                'embedding_dir': "./embeddings",
                'concept_dir': "./concepts",
                'embeddings': {
                    'hair_colors': [
                        "blue_hair", "red_hair", "blonde_hair", "pink_hair", "green_hair",
                        "purple_hair", "white_hair", "black_hair"
                    ],
                    'eye_colors': [
                        "blue_eyes", "red_eyes", "green_eyes", "yellow_eyes",
                        "purple_eyes", "heterochromia"
                    ],
                    'character_types': [
                        "tsundere", "kuudere", "yandere", "dandere", "deredere",
                        "himedere", "otaku", "protagonist", "villain"
                    ]
                }
            }
        }
        
        # Write model config
        with open(model_config_file, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
    else:
        print(f"Model config file {model_config_file} already exists.")
    
    # Create required directories
    dirs_to_create = [
        "models",
        "outputs",
        "cache",
        "data",
        "temp",
        "embeddings",
        "concepts",
        "data/style_references",
        "outputs/images",
        "outputs/animations",
        "outputs/voices",
        "outputs/models",
        "outputs/characters",
        "outputs/lineage"
    ]
    
    for dir_path in dirs_to_create:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Configuration setup complete!")
    return True

if __name__ == "__main__":
    ensure_config_files()