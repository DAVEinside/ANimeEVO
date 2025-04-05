"""
Model utilities for the anime character evolution system.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import json
import yaml
import requests
import tempfile
import hashlib
import time
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)

def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return information.
    
    Returns:
        Dictionary with GPU information
    """
    try:
        if not torch.cuda.is_available():
            return {
                'available': False,
                'reason': 'CUDA not available',
                'device_count': 0
            }
            
        device_count = torch.cuda.device_count()
        devices = []
        
        for i in range(device_count):
            try:
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                
                device_info = {
                    'index': i,
                    'name': device_name,
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'total_memory_mb': device_props.total_memory // (1024 * 1024),
                    'multi_processor_count': device_props.multi_processor_count
                }
                
                devices.append(device_info)
            except Exception as e:
                logger.error(f"Error getting device {i} info: {e}")
                devices.append({
                    'index': i,
                    'name': 'Unknown',
                    'error': str(e)
                })
                
        # Get CUDA version
        cuda_version = torch.version.cuda
        
        return {
            'available': True,
            'device_count': device_count,
            'cuda_version': cuda_version,
            'devices': devices,
            'pytorch_version': torch.__version__
        }
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return {
            'available': False,
            'reason': str(e),
            'device_count': 0
        }

def get_optimal_device() -> torch.device:
    """
    Get the optimal device for running models.
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        # Find the GPU with the most free memory
        device_count = torch.cuda.device_count()
        max_free_memory = 0
        best_device = 0
        
        for i in range(device_count):
            try:
                # Get free memory
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_memory = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = i
            except Exception:
                continue
                
        return torch.device(f"cuda:{best_device}")
    else:
        return torch.device("cpu")

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize a PyTorch model for inference.
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # Use torch.inference_mode for forward passes
    original_forward = model.forward
    
    def inference_forward(*args, **kwargs):
        with torch.inference_mode():
            return original_forward(*args, **kwargs)
            
    model.forward = inference_forward
    
    # Try to use torch.jit.script for optimization if possible
    try:
        model = torch.jit.script(model)
    except Exception as e:
        logger.warning(f"Could not script model: {e}")
    
    return model

def download_file(url: str, destination: str, chunk_size: int = 8192) -> bool:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download
        destination: Destination path
        chunk_size: Chunk size for downloading
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        
        # Stream the download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Print download progress
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination))
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
                    
        progress_bar.close()
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def download_model(
    model_info: Dict[str, Any],
    models_dir: str = "models",
    check_hash: bool = True
) -> Optional[str]:
    """
    Download a model from a URL.
    
    Args:
        model_info: Dictionary with model information
        models_dir: Directory to save the model
        check_hash: Whether to check the file hash
        
    Returns:
        Path to the downloaded model or None if failed
    """
    try:
        if 'url' not in model_info:
            logger.error("Model info missing URL")
            return None
            
        url = model_info['url']
        filename = model_info.get('filename', os.path.basename(url))
        model_hash = model_info.get('hash')
        hash_type = model_info.get('hash_type', 'md5')
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Define destination path
        destination = os.path.join(models_dir, filename)
        
        # Check if file already exists and has correct hash
        if os.path.exists(destination) and check_hash and model_hash:
            if verify_file_hash(destination, model_hash, hash_type):
                logger.info(f"Model already exists and hash matches: {destination}")
                return destination
                
        # Download the file
        logger.info(f"Downloading model from {url} to {destination}")
        if not download_file(url, destination):
            return None
            
        # Verify hash if provided
        if check_hash and model_hash:
            if not verify_file_hash(destination, model_hash, hash_type):
                logger.error(f"Hash verification failed for {destination}")
                os.remove(destination)
                return None
                
        return destination
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def verify_file_hash(filepath: str, expected_hash: str, hash_type: str = 'md5') -> bool:
    """
    Verify a file's hash.
    
    Args:
        filepath: Path to the file
        expected_hash: Expected hash
        hash_type: Hash type (md5, sha1, sha256)
        
    Returns:
        True if hash matches, False otherwise
    """
    try:
        if hash_type == 'md5':
            hasher = hashlib.md5()
        elif hash_type == 'sha1':
            hasher = hashlib.sha1()
        elif hash_type == 'sha256':
            hasher = hashlib.sha256()
        else:
            logger.error(f"Unsupported hash type: {hash_type}")
            return False
            
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
                
        calculated_hash = hasher.hexdigest()
        
        return calculated_hash.lower() == expected_hash.lower()
    except Exception as e:
        logger.error(f"Error verifying file hash: {e}")
        return False

def create_model_info_file(
    model_path: str,
    info: Dict[str, Any],
    basename: Optional[str] = None
) -> Optional[str]:
    """
    Create a model info file.
    
    Args:
        model_path: Path to the model
        info: Model information
        basename: Base name for the info file (defaults to model filename)
        
    Returns:
        Path to the info file or None if failed
    """
    try:
        if basename is None:
            basename = os.path.basename(model_path)
            
        info_path = os.path.join(os.path.dirname(model_path), f"{basename}.info.json")
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        return info_path
    except Exception as e:
        logger.error(f"Error creating model info file: {e}")
        return None

def load_model_info(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Load model info from an info file.
    
    Args:
        model_path: Path to the model or info file
        
    Returns:
        Model information or None if not found
    """
    try:
        # Try model.info.json first
        if model_path.endswith('.info.json'):
            info_path = model_path
        else:
            basename = os.path.basename(model_path)
            info_path = os.path.join(os.path.dirname(model_path), f"{basename}.info.json")
            
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
                
        # Try model.yaml as fallback
        yaml_path = os.path.join(os.path.dirname(model_path), f"{basename}.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
                
        return None
    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        return None

def measure_inference_time(
    model: torch.nn.Module,
    sample_input: Union[torch.Tensor, Tuple, List],
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Measure model inference time.
    
    Args:
        model: PyTorch model
        sample_input: Sample input for the model
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        device: Device to run on (defaults to model's device)
        
    Returns:
        Dictionary with timing statistics
    """
    if device is None:
        # Try to determine device from model
        device = next(model.parameters()).device
        
    # Ensure model is in eval mode
    model.eval()
    
    # Move sample input to device
    if isinstance(sample_input, torch.Tensor):
        sample_input = sample_input.to(device)
    elif isinstance(sample_input, tuple):
        sample_input = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in sample_input)
    elif isinstance(sample_input, list):
        sample_input = [x.to(device) if isinstance(x, torch.Tensor) else x for x in sample_input]
        
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(sample_input)
            
    # Measurement runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    run_times = []
    
    for _ in range(num_runs):
        # Synchronize CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        # Start timing
        if device.type == 'cuda':
            start_event.record()
            start_time = time.time()
        else:
            start_time = time.time()
            
        # Run inference
        with torch.no_grad():
            _ = model(sample_input)
            
        # End timing
        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            run_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        else:
            run_time = time.time() - start_time
            
        run_times.append(run_time)
        
    # Calculate statistics
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    min_time = np.min(run_times)
    max_time = np.max(run_times)
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'runs': num_runs,
        'device': str(device)
    }

def get_optimal_batch_size(
    model: torch.nn.Module,
    sample_input_fn: Callable[[int], torch.Tensor],
    device: torch.device = None,
    starting_size: int = 1,
    max_size: int = 32,
    target_memory_usage: float = 0.7,
    safety_margin: float = 0.9
) -> int:
    """
    Determine the optimal batch size for a model.
    
    Args:
        model: PyTorch model
        sample_input_fn: Function that takes batch size and returns sample input
        device: Device to use (defaults to model's device)
        starting_size: Starting batch size
        max_size: Maximum batch size to try
        target_memory_usage: Target memory usage ratio (0-1)
        safety_margin: Safety margin for the result (0-1)
        
    Returns:
        Optimal batch size
    """
    if device is None:
        # Try to determine device from model
        device = next(model.parameters()).device
        
    if device.type != 'cuda':
        # For CPU, we don't need to check memory
        return max_size
        
    # Ensure model is in eval mode
    model.eval()
    model.to(device)
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = total_memory * target_memory_usage
    
    # Start with an initial batch size
    batch_size = starting_size
    
    while batch_size <= max_size:
        try:
            # Reset GPU memory
            torch.cuda.empty_cache()
            
            # Get sample input for this batch size
            sample_input = sample_input_fn(batch_size).to(device)
            
            # Run a forward pass
            with torch.no_grad():
                _ = model(sample_input)
                
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(device)
            
            if memory_used > target_memory:
                break
                
            # Try the next batch size
            batch_size *= 2
            
        except RuntimeError as e:
            # Probably out of memory
            logger.info(f"Error with batch size {batch_size}: {e}")
            break
            
    # Binary search for the optimal batch size
    lower_bound = batch_size // 2
    upper_bound = batch_size
    
    while lower_bound < upper_bound - 1:
        mid = (lower_bound + upper_bound) // 2
        
        try:
            # Reset GPU memory
            torch.cuda.empty_cache()
            
            # Get sample input for this batch size
            sample_input = sample_input_fn(mid).to(device)
            
            # Run a forward pass
            with torch.no_grad():
                _ = model(sample_input)
                
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(device)
            
            if memory_used > target_memory:
                upper_bound = mid
            else:
                lower_bound = mid
                
        except RuntimeError:
            # Probably out of memory
            upper_bound = mid
            
    # Apply safety margin
    optimal_batch_size = max(1, int(lower_bound * safety_margin))
    
    return optimal_batch_size

def quantize_model(
    model: torch.nn.Module,
    quantization_method: str = 'dynamic',
    dtype: torch.dtype = torch.qint8
) -> torch.nn.Module:
    """
    Quantize a PyTorch model for improved performance.
    
    Args:
        model: PyTorch model
        quantization_method: Quantization method ('static', 'dynamic', 'qat')
        dtype: Quantization data type
        
    Returns:
        Quantized model
    """
    try:
        model.eval()
        
        if quantization_method == 'dynamic':
            quantized_model = torch.quantization.quantize_dynamic(
                model, qconfig_spec={torch.nn.Linear}, dtype=dtype
            )
        elif quantization_method == 'static':
            # This requires calibration data, which isn't provided here
            # For a real implementation, you would need to provide a calibration data loader
            logger.warning("Static quantization requires calibration data. Using dynamic quantization instead.")
            quantized_model = torch.quantization.quantize_dynamic(
                model, qconfig_spec={torch.nn.Linear}, dtype=dtype
            )
        elif quantization_method == 'qat':
            logger.warning("Quantization-aware training requires training. Using dynamic quantization instead.")
            quantized_model = torch.quantization.quantize_dynamic(
                model, qconfig_spec={torch.nn.Linear}, dtype=dtype
            )
        else:
            logger.warning(f"Unknown quantization method: {quantization_method}. Using dynamic quantization.")
            quantized_model = torch.quantization.quantize_dynamic(
                model, qconfig_spec={torch.nn.Linear}, dtype=dtype
            )
            
        return quantized_model
    except Exception as e:
        logger.error(f"Error quantizing model: {e}")
        logger.warning("Returning original model")
        return model

def optimize_memory_usage():
    """
    Optimize memory usage for PyTorch.
    """
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Garbage collect
    import gc
    gc.collect()
    
    # Use more efficient memory allocator if available
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.reset_peak_memory_stats()

def get_model_stats(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get statistics about a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model statistics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in memory
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Model size in MB
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Get model structure
    model_structure = str(model)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'model_structure': model_structure
    }

def find_memory_leaks(
    model: torch.nn.Module,
    sample_input_fn: Callable[[int], torch.Tensor],
    iterations: int = 10
) -> Dict[str, Any]:
    """
    Find memory leaks in a model.
    
    Args:
        model: PyTorch model
        sample_input_fn: Function that returns sample input
        iterations: Number of iterations to run
        
    Returns:
        Dictionary with memory usage statistics
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
        
    try:
        # Get device
        device = next(model.parameters()).device
        
        # Ensure model is in eval mode
        model.eval()
        
        # Initial memory usage
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated(device)
        
        # Run iterations
        memory_usage = [start_memory]
        
        for _ in range(iterations):
            # Get sample input
            sample_input = sample_input_fn(1).to(device)
            
            # Run inference
            with torch.no_grad():
                _ = model(sample_input)
                
            # Record memory usage
            current_memory = torch.cuda.memory_allocated(device)
            memory_usage.append(current_memory)
            
        # Final cleanup
        torch.cuda.empty_cache()
        end_memory = torch.cuda.memory_allocated(device)
        
        # Calculate statistics
        initial_memory_mb = start_memory / (1024 * 1024)
        final_memory_mb = end_memory / (1024 * 1024)
        peak_memory_mb = max(memory_usage) / (1024 * 1024)
        
        memory_leak_mb = final_memory_mb - initial_memory_mb
        
        # Return statistics
        return {
            'initial_memory_mb': initial_memory_mb,
            'final_memory_mb': final_memory_mb,
            'peak_memory_mb': peak_memory_mb,
            'memory_leak_mb': memory_leak_mb,
            'memory_usage_mb': [m / (1024 * 1024) for m in memory_usage],
            'has_leak': memory_leak_mb > 1.0  # Consider >1MB as a leak
        }
    except Exception as e:
        logger.error(f"Error finding memory leaks: {e}")
        return {'error': str(e)}