from src.ui.gpu_governor_ui import GPUGovernorUI
import logging
import sys

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('gpu_governor.log')
        ]
    )

def main():
    """Main entry point for the GPU Governor application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting NVIDIA GPU Governor")
        ui = GPUGovernorUI()
        ui.launch()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()