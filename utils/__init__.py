"""
Utilities package for the Anime Character Evolution System.

This package provides utility functions:
- image_processing: Image manipulation utilities
- data_handling: Data loading and saving utilities
- model_utils: Utilities for model management and optimization
"""

from .image_processing import (
    resize_image, crop_image, add_overlay_text, create_image_grid,
    apply_image_effects, image_to_base64, base64_to_image
)

from .data_handling import (
    load_json, save_json, load_yaml, save_yaml, load_csv, save_csv,
    load_pickle, save_pickle, list_files, ensure_directory, delete_file,
    file_exists, get_file_info, clean_directory, find_by_id_prefix,
    generate_unique_id, generate_timestamp, load_config, save_config
)

from .model_utils import (
    check_gpu_availability, get_optimal_device, set_seed, optimize_for_inference,
    download_file, download_model, verify_file_hash, measure_inference_time,
    get_optimal_batch_size, quantize_model, optimize_memory_usage
)

__all__ = [
    # Image processing utils
    'resize_image', 'crop_image', 'add_overlay_text', 'create_image_grid',
    'apply_image_effects', 'image_to_base64', 'base64_to_image',
    
    # Data handling utils
    'load_json', 'save_json', 'load_yaml', 'save_yaml', 'load_csv', 'save_csv',
    'load_pickle', 'save_pickle', 'list_files', 'ensure_directory', 'delete_file',
    'file_exists', 'get_file_info', 'clean_directory', 'find_by_id_prefix',
    'generate_unique_id', 'generate_timestamp', 'load_config', 'save_config',
    
    # Model utils
    'check_gpu_availability', 'get_optimal_device', 'set_seed', 'optimize_for_inference',
    'download_file', 'download_model', 'verify_file_hash', 'measure_inference_time',
    'get_optimal_batch_size', 'quantize_model', 'optimize_memory_usage'
]