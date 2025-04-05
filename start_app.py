"""
Startup script for the Anime Character Evolution System.
Initializes configuration and starts the web application.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('anime_evolution.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Initialize and start the application."""
    try:
        # Ensure we're in the project root directory
        project_root = Path(__file__).parent.absolute()
        os.chdir(project_root)
        
        # Add project root to system path
        sys.path.insert(0, str(project_root))
        
        logger.info("Starting Anime Character Evolution System...")
        
        # Initialize configuration
        logger.info("Initializing configuration...")
        from init_config import ensure_config_files
        ensure_config_files()
        
        # Start the web application
        logger.info("Starting web application...")
        from interface.web_app import AnimeEvolutionWebApp
        app = AnimeEvolutionWebApp()
        app.run()
        
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())