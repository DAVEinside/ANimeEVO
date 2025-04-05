"""
Configuration package for the Anime Character Evolution System.
"""

import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from the specified YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return {}
        
def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.
    
    Returns:
        Dictionary containing default configuration
    """
    return {
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
        }
    }