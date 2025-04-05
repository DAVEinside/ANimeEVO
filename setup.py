"""
Complete setup script for Anime Character Evolution System.
This script will:
1. Install required dependencies
2. Create/update configuration files
3. Set up directory structure
4. Create required static files
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command):
    """Run a command and print output."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
        
    process.wait()
    return process.returncode

def install_dependencies():
    """Install required Python dependencies."""
    print("\n====== Installing Dependencies ======")
    dependencies = [
        "pillow",       # For image processing
        "torch",        # PyTorch
        "diffusers",    # Diffusion models
        "transformers", # Transformers
        "numpy",        # Numerical computing
        "flask",        # Web server
        "pyyaml",       # YAML parsing
        "tqdm",         # Progress bars
        "requests",     # HTTP requests
        "safetensors"   # For loading .safetensors models
    ]
    
    try:
        for dep in dependencies:
            result = run_command(f"{sys.executable} -m pip install {dep}")
            if result != 0:
                print(f"Warning: Failed to install {dep}")
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False
        
    return True

def create_directory_structure():
    """Create required directory structure."""
    print("\n====== Creating Directory Structure ======")
    dirs = [
        "models",
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
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
    
    # Also create static directories
    try:
        from create_static_dirs import create_static_dirs
        create_static_dirs()
    except ImportError:
        print("Running create_static_dirs script directly...")
        run_command(f"{sys.executable} create_static_dirs.py")
    
    return True

def update_configuration():
    """Update configuration to match model structure."""
    print("\n====== Updating Configuration ======")
    try:
        from update_config import update_model_config
        update_model_config()
    except ImportError:
        print("Running update_config script directly...")
        run_command(f"{sys.executable} update_config.py")
    
    return True

def main():
    """Main setup function."""
    print("====== Anime Character Evolution System Setup ======")
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Setup incomplete.")
        return 1
    
    # Create directory structure
    if not create_directory_structure():
        print("Failed to create directory structure. Setup incomplete.")
        return 1
    
    # Update configuration
    if not update_configuration():
        print("Failed to update configuration. Setup incomplete.")
        return 1
    
    print("\n====== Setup Complete! ======")
    print("You can now run the application with:")
    print("  python start_app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())