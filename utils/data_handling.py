"""
Data handling utilities for the anime character evolution system.
"""

import os
import json
import yaml
import csv
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import uuid

# Setup logging
logger = logging.getLogger(__name__)

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return {}

def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
        indent: Indentation level for pretty printing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False

def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load data from a YAML file.
    
    Args:
        filepath: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML from {filepath}: {e}")
        return {}

def save_yaml(data: Dict[str, Any], filepath: str) -> bool:
    """
    Save data to a YAML file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the YAML file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"Error saving YAML to {filepath}: {e}")
        return False

def load_csv(filepath: str, as_dict: bool = True) -> Union[List[Dict[str, Any]], List[List[str]]]:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        as_dict: Whether to return a list of dictionaries (True) or a list of lists (False)
        
    Returns:
        List of dictionaries or list of lists containing the CSV data
    """
    try:
        with open(filepath, 'r', encoding='utf-8', newline='') as f:
            if as_dict:
                reader = csv.DictReader(f)
                return list(reader)
            else:
                reader = csv.reader(f)
                return list(reader)
    except Exception as e:
        logger.error(f"Error loading CSV from {filepath}: {e}")
        return []

def save_csv(data: Union[List[Dict[str, Any]], List[List[str]]], filepath: str, fieldnames: List[str] = None) -> bool:
    """
    Save data to a CSV file.
    
    Args:
        data: List of dictionaries or list of lists to save
        filepath: Path to save the CSV file
        fieldnames: List of field names (required if data is a list of lists)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            if data and isinstance(data[0], dict):
                # List of dictionaries
                if fieldnames is None:
                    fieldnames = list(data[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            else:
                # List of lists
                writer = csv.writer(f)
                if fieldnames:
                    writer.writerow(fieldnames)
                writer.writerows(data)
        return True
    except Exception as e:
        logger.error(f"Error saving CSV to {filepath}: {e}")
        return False

def load_pickle(filepath: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Unpickled data
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle from {filepath}: {e}")
        return None

def save_pickle(data: Any, filepath: str) -> bool:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to pickle
        filepath: Path to save the pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving pickle to {filepath}: {e}")
        return False

def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> List[str]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match files
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    try:
        path = Path(directory)
        if recursive:
            return [str(p) for p in path.glob(f"**/{pattern}") if p.is_file()]
        else:
            return [str(p) for p in path.glob(pattern) if p.is_file()]
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        return []

def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        True if the directory exists or was created, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False

def delete_file(filepath: str) -> bool:
    """
    Delete a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        True if successful or file doesn't exist, False otherwise
    """
    try:
        path = Path(filepath)
        if path.exists():
            path.unlink()
        return True
    except Exception as e:
        logger.error(f"Error deleting file {filepath}: {e}")
        return False

def file_exists(filepath: str) -> bool:
    """
    Check if a file exists.
    
    Args:
        filepath: Path to the file
        
    Returns:
        True if the file exists, False otherwise
    """
    return Path(filepath).exists()

def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return {}
            
        stats = path.stat()
        return {
            'name': path.name,
            'extension': path.suffix,
            'size': stats.st_size,
            'created': datetime.fromtimestamp(stats.st_ctime),
            'modified': datetime.fromtimestamp(stats.st_mtime),
            'is_directory': path.is_dir()
        }
    except Exception as e:
        logger.error(f"Error getting file info for {filepath}: {e}")
        return {}

def clean_directory(directory: str, pattern: str = "*", max_age_days: Optional[int] = None) -> int:
    """
    Clean a directory by deleting files matching a pattern and/or older than a certain age.
    
    Args:
        directory: Directory to clean
        pattern: Glob pattern to match files
        max_age_days: Maximum age of files to keep in days (None to keep all)
        
    Returns:
        Number of files deleted
    """
    try:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return 0
            
        files = path.glob(pattern)
        now = datetime.now()
        
        count = 0
        for file_path in files:
            if file_path.is_file():
                # Check age if max_age_days is specified
                if max_age_days is not None:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    age_days = (now - mtime).days
                    if age_days <= max_age_days:
                        continue
                
                # Delete file
                file_path.unlink()
                count += 1
                
        return count
    except Exception as e:
        logger.error(f"Error cleaning directory {directory}: {e}")
        return 0

def find_by_id_prefix(directory: str, id_prefix: str, extension: str = ".json") -> Optional[str]:
    """
    Find a file by ID prefix.
    
    Args:
        directory: Directory to search
        id_prefix: ID prefix to match
        extension: File extension to match
        
    Returns:
        Path to the first matching file, or None if not found
    """
    try:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return None
            
        for file_path in path.glob(f"*{id_prefix}*{extension}"):
            if file_path.is_file():
                return str(file_path)
                
        return None
    except Exception as e:
        logger.error(f"Error finding file by ID prefix in {directory}: {e}")
        return None

def batch_process_files(
    directory: str,
    process_func: callable,
    pattern: str = "*.json",
    recursive: bool = False,
    max_files: Optional[int] = None
) -> Tuple[int, int]:
    """
    Batch process files in a directory.
    
    Args:
        directory: Directory to process
        process_func: Function to call for each file (takes filepath as argument)
        pattern: Glob pattern to match files
        recursive: Whether to search recursively
        max_files: Maximum number of files to process (None for all)
        
    Returns:
        Tuple of (number of files processed, number of files failed)
    """
    try:
        files = list_files(directory, pattern, recursive)
        
        if max_files is not None:
            files = files[:max_files]
            
        success_count = 0
        fail_count = 0
        
        for filepath in files:
            try:
                result = process_func(filepath)
                if result:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"Error processing file {filepath}: {e}")
                fail_count += 1
                
        return success_count, fail_count
    except Exception as e:
        logger.error(f"Error batch processing files in {directory}: {e}")
        return 0, 0

def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id

def generate_timestamp() -> str:
    """
    Generate a formatted timestamp.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file (YAML or JSON).
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        ext = os.path.splitext(config_path)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            return load_yaml(config_path)
        elif ext in ['.json']:
            return load_json(config_path)
        else:
            logger.warning(f"Unknown config file extension: {ext}. Trying YAML first, then JSON.")
            config = load_yaml(config_path)
            if not config:
                config = load_json(config_path)
            return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to a file (YAML or JSON).
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ext = os.path.splitext(config_path)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            return save_yaml(config, config_path)
        elif ext in ['.json']:
            return save_json(config, config_path)
        else:
            logger.warning(f"Unknown config file extension: {ext}. Saving as YAML.")
            return save_yaml(config, config_path)
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        return False

def save_numpy_array(array: np.ndarray, filepath: str) -> bool:
    """
    Save a NumPy array to a file.
    
    Args:
        array: NumPy array to save
        filepath: Path to save the array
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        np.save(filepath, array)
        return True
    except Exception as e:
        logger.error(f"Error saving NumPy array to {filepath}: {e}")
        return False

def load_numpy_array(filepath: str) -> Optional[np.ndarray]:
    """
    Load a NumPy array from a file.
    
    Args:
        filepath: Path to the array file
        
    Returns:
        NumPy array or None if an error occurred
    """
    try:
        return np.load(filepath)
    except Exception as e:
        logger.error(f"Error loading NumPy array from {filepath}: {e}")
        return None

def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import psutil
    
    try:
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent,
            'hostname': platform.node()
        }
        
        # Add GPU info if available
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_version'] = torch.version.cuda
        except ImportError:
            info['cuda_available'] = False
            
        return info
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {'error': str(e)}