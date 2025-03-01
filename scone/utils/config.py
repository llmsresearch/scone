"""Configuration utilities for SCONE."""

import os
from typing import Dict, Any, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file.
    
    Returns:
        Dictionary containing the configuration.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Dictionary containing the configuration.
        config_path: Path to save the configuration file.
    
    Raises:
        yaml.YAMLError: If the configuration cannot be serialized to YAML.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration.
        override_config: Configuration to override base_config.
    
    Returns:
        Merged configuration.
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config


def get_config(
    config_path: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get configuration from a file and/or override with provided config.
    
    Args:
        config_path: Path to the configuration file.
        override_config: Configuration to override the loaded configuration.
    
    Returns:
        Dictionary containing the configuration.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file is not valid YAML.
    """
    if config_path is not None:
        config = load_config(config_path)
    else:
        config = {}
    
    if override_config is not None:
        config = merge_configs(config, override_config)
    
    return config 