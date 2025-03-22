"""
Configuration management for the trading project
"""

import os
import json
import yaml
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for the trading project"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file or defaults
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        # Default configuration
        self.config = {
            "data": {
                "data_file": "USATECH.IDXUSD_Candlestick_15_M_BID_01.01.2023-18.01.2025.csv",
                "test_size": 0.2
            },
            "model": {
                "type": "random_forest",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 10,
                "random_state": 42
            },
            "trading": {
                "profit_target": 0.01,
                "stop_loss": 0.005,
                "future_periods": 10,
                "confidence_threshold": 0.6
            },
            "output": {
                "results_dir": "results",
                "save_model": True,
                "save_plots": True
            }
        }
        
        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key (str): Dot-separated key (e.g., 'model.n_estimators')
            default (any, optional): Default value if key not found
            
        Returns:
            any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key (str): Dot-separated key (e.g., 'model.n_estimators')
            value (any): Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary
        
        Args:
            config_dict (dict): Dictionary to update config with
        """
        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = _update(self.config, config_dict)
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file
        
        Args:
            config_path (str): Path to configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif ext.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
        
        self.update(config_dict)
    
    def save_config(self, config_path: str) -> None:
        """
        Save configuration to file
        
        Args:
            config_path (str): Path to save configuration to
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif ext.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required to save YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
    
    def __str__(self) -> str:
        """String representation of config"""
        return json.dumps(self.config, indent=2)


# Create a global config instance
config = Config()
