import logging
import logging.config
import os
from utils.config_manager import ConfigManager

def setup_logging():
    """
    Setup logging configuration.
    """
    config_manager = ConfigManager()
    logging_config = config_manager.get_logging_config()

    # Create logs directory if it doesn't exist
    log_file_path = logging_config.get('handlers', {}).get('file', {}).get('filename')
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig(logging_config)
