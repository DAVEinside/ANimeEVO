import os
from utils.model_utils import download_model
from utils.data_handling import ensure_directory  # Correct import


# Create models directory
models_dir = "models"
ensure_directory(models_dir)

# Alternative anime models/LoRAs that are publicly available
anime_models = [
    {
        "name": "Anime Style LoRA",
        "url": "https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0-pruned.safetensors",
        "filename": "anything-v3.0.safetensors",
        "type": "diffusion"
    },
    {
        "name": "Anime Character LoRA",
        "url": "https://huggingface.co/hakurei/waifu-diffusion/resolve/main/wd-1-5-beta2.safetensors",
        "filename": "waifu-diffusion-1-5.safetensors",
        "type": "diffusion"
    }
]

# Download each model
for model_info in anime_models:
    print(f"Downloading {model_info['name']}...")
    download_model(model_info, models_dir)
    print(f"Downloaded {model_info['name']} to {models_dir}/{model_info['filename']}")

print("Anime model downloads complete!")