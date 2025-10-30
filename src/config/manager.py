"""
Configuration management for saving/loading analysis settings.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages saving and loading of analysis configurations.

    Configurations are saved as YAML files for human readability.
    """

    def __init__(self, config_dir: Path = Path("configs")):
        """
        Initialize config manager.

        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        logger.info(f"ConfigManager initialized with directory: {self.config_dir}")

    def save_config(
        self,
        name: str,
        filters: dict,
        regression: dict,
        kfold: dict,
        outlier: dict,
        description: str = "",
    ) -> Path:
        """
        Save analysis configuration to YAML file.

        Args:
            name: Configuration name (will be used as filename)
            filters: Filter configuration
            regression: Regression model configuration
            kfold: K-Fold configuration
            outlier: Outlier detection configuration
            description: Optional description

        Returns:
            Path to saved configuration file
        """
        config = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "filters": filters,
            "regression": regression,
            "kfold": kfold,
            "outlier": outlier,
        }

        # Sanitize filename
        safe_name = "".join(
            c for c in name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_name = safe_name.replace(" ", "_")

        filepath = self.config_dir / f"{safe_name}.yaml"

        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {filepath}")
        return filepath

    def load_config(self, filepath: Path) -> dict:
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to configuration file

        Returns:
            Configuration dictionary
        """
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from: {filepath}")
        return config

    def list_configs(self) -> list[Path]:
        """
        List all available configuration files.

        Returns:
            List of configuration file paths
        """
        return sorted(self.config_dir.glob("*.yaml"))

    def get_config_info(self, filepath: Path) -> dict:
        """
        Get basic info about a configuration without loading it fully.

        Args:
            filepath: Path to configuration file

        Returns:
            Dictionary with name, description, created_at
        """
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        return {
            "name": config.get("name", filepath.stem),
            "description": config.get("description", ""),
            "created_at": config.get("created_at", "Unknown"),
            "filepath": filepath,
        }

    def delete_config(self, filepath: Path) -> None:
        """Delete a configuration file."""
        filepath.unlink()
        logger.info(f"Configuration deleted: {filepath}")
