"""
Configuration utilities for Neural Retrieval Dynamics (NRD)
This module provides functions for loading and saving configuration files.
"""

import yaml
import os
from typing import Dict, Any, Optional
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle defaults
    if 'defaults' in config:
        default_config = config.pop('defaults')
        if isinstance(default_config, list):
            for default in default_config:
                if isinstance(default, str):
                    # Load default config
                    default_path = os.path.join(os.path.dirname(config_path), default + '.yaml')
                    if os.path.exists(default_path):
                        base_config = load_config(default_path)
                        # Merge configs (current config overrides base)
                        config = merge_configs(base_config, config)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration (takes precedence)
        
    Returns:
        Dict: Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def update_config_from_args(config: Dict, args: Dict) -> Dict:
    """
    Update configuration from command-line arguments.
    
    Args:
        config: Base configuration
        args: Dictionary of command-line arguments
        
    Returns:
        Dict: Updated configuration
    """
    for key, value in args.items():
        if value is not None:
            # Handle nested keys (e.g., 'training.learning_rate')
            keys = key.split('.')
            current = config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
    
    return config


def get_config_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get a value from configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'training.learning_rate')
        default: Default value if key not found
        
    Returns:
        Any: Configuration value
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if key in current:
            current = current[key]
        else:
            return default
    
    return current
