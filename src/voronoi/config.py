"""Configuration management for the Voronoi diagram generator."""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the Voronoi diagram generator."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config_data = self._load_config()

    def _find_config_file(self) -> str:
        """Find the configuration file."""
        # Look for config.json in the project root
        current_dir = Path(__file__).parent.parent.parent
        config_file = current_dir / "config.json"

        if config_file.exists():
            return str(config_file)

        # Fallback to a default config
        return self._create_default_config()

    def _create_default_config(self) -> str:
        """Create a default configuration file."""
        default_config = {
            "visualization": {
                "show_sites": True,
                "show_vertices": False,
                "show_edges": False,
                "show_faces": True,
                "show_grid": False,
                "show_bounding_box": False,
                "show_stats": True
            },
            "rendering": {
                "figure_size": [12, 10],
                "dpi": 100,
                "face_alpha": 1.0,
                "edge_alpha": 0.8,
                "site_alpha": 0.3,
                "vertex_alpha": 0.9,
                "anti_aliasing": True,
                "high_quality": True
            }
        }

        config_path = Path(__file__).parent.parent.parent / "config.json"
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        return str(config_path)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config file {self.config_path}: {e}")
            return {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the config value (e.g., "visualization.show_edges")
            default: Default value if key is not found

        Returns:
            The configuration value or default
        """
        keys = key_path.split('.')
        value = self._config_data

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config_data

        # Navigate to the parent dict
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

    def save(self) -> None:
        """Save the current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self._config_data, f, indent=2)
        except IOError as e:
            print(f"Error saving config file {self.config_path}: {e}")

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config_data = self._load_config()

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self._config_data.get(section, {})

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update an entire configuration section."""
        if section not in self._config_data:
            self._config_data[section] = {}
        self._config_data[section].update(values)


# Global configuration instance
_config_instance = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def reload_config() -> None:
    """Reload the global configuration."""
    global _config_instance
    if _config_instance:
        _config_instance.reload()
    else:
        _config_instance = Config()