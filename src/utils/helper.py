import os
import logging
import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root directory."""
    # This file is located in src/utils/helper.py, so project root is 2 levels up
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = None) -> dict:
    """Loads configuration from config.yaml."""
    if config_path is None:
        root = get_project_root()
        config_path = os.path.join(root, "configs", "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with stdout handler and standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
