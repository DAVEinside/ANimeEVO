"""
Script to update configuration to match the user's specific model structure.
"""

import os
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def update_model_config():
    """Update configuration to match user's specific model setup."""
    logger.info("Updating configuration for your specific model structure...")
    
    # First make sure config directory exists
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Main config file
    config_file = config_dir / "config.yaml"
    
    # Check if config file exists, create/update it
    config = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Error reading existing config: {e}")
    
    # Update with specific diffusion configuration
    config['diffusion'] = {
        'base_model': "models/diffusion/waiNSFWillustrious_v130.safetensors",
        'anime_model': "models/diffusion/waiNSFWillustrious_v130.safetensors",
        'inference': {
            'steps': 30,
            'guidance_scale': 7.5,
            'width': 512,
            'height': 512,
            'sampler': 'DDIM',
            'clip_skip': 1,
            'seed': -1
        }
    }
    
    # Make sure paths section exists
    if 'paths' not in config:
        config['paths'] = {}
    
    # Update paths
    config['paths'].update({
        'models_dir': "./models",
        'output_dir': "./outputs",
        'cache_dir': "./cache",
        'data_dir': "./data",
        'temp_dir': "./temp"
    })
    
    # Update interface if missing
    if 'interface' not in config:
        config['interface'] = {
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
        }
    
    # Write updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Model config file
    model_config_file = config_dir / "model_config.yaml"
    
    # Check if model config file exists, create/update it
    model_config = {}
    if model_config_file.exists():
        try:
            with open(model_config_file, 'r') as f:
                model_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Error reading existing model config: {e}")
    
    # Update with specific model configuration
    model_config['diffusion'] = {
        'base_model': "models/diffusion/waiNSFWillustrious_v130.safetensors",
        'anime_model': "models/diffusion/waiNSFWillustrious_v130.safetensors",
        'anime_lora': "models/lora/realisticVision6081_v51H",
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
        }
    }
    
    # Add LoRA configurations
    if 'lora' not in model_config:
        model_config['lora'] = {}
    
    model_config['lora']['models'] = [
        {
            'name': "realistic_vision",
            'path': "models/lora/realisticVision6081_v51H",
            'alpha': 0.7
        },
        {
            'name': "pruned_model",
            'path': "models/lora/v1-5-pruned.safetensors",
            'alpha': 0.7
        }
    ]
    
    # Write updated model config
    with open(model_config_file, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)
    
    logger.info(f"Updated configuration files:")
    logger.info(f"  - {config_file}")
    logger.info(f"  - {model_config_file}")
    logger.info("Configuration updated successfully to match your model structure.")
    
    # Create required directories
    dirs_to_create = [
        "outputs",
        "cache",
        "data",
        "temp",
        "outputs/images",
        "outputs/animations",
        "outputs/voices",
        "outputs/characters",
        "outputs/lineage"
    ]
    
    for dir_path in dirs_to_create:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return True

if __name__ == "__main__":
    update_model_config()
    print("\nYour configuration has been updated to match your model structure.")
    print("Now run 'python start_app.py' to start the application.")