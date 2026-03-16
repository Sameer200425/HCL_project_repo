"""
Logging utilities for training, evaluation, and experiment tracking.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "bank_vit",
    log_dir: str = "logs/",
    level: str = "INFO",
) -> logging.Logger:
    """
    Create and configure a logger with file and console handlers.
    
    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    
    Returns:
        Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}.log"),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class TrainingHistory:
    """Track and persist training metrics across epochs."""
    
    def __init__(self, save_dir: str = "logs/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "val_roc_auc": [],
            "learning_rate": [],
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Add metrics for a completed epoch."""
        self.history["epoch"].append(epoch)
        for key in self.history:
            if key != "epoch" and key in metrics:
                self.history[key].append(metrics[key])
    
    def save(self, filename: str = "training_history.json") -> str:
        """Save history to JSON file."""
        path = self.save_dir / filename
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return str(path)
    
    def load(self, filename: str = "training_history.json") -> Dict:
        """Load history from JSON file."""
        path = self.save_dir / filename
        with open(path, "r") as f:
            self.history = json.load(f)
        return self.history


class HyperparameterLogger:
    """Log and persist hyperparameters for experiment reproducibility."""
    
    def __init__(self, save_dir: str = "logs/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None,
    ) -> str:
        """
        Save hyperparameters to JSON.
        
        Args:
            config: Dictionary of hyperparameters.
            experiment_name: Optional name for the experiment.
        
        Returns:
            Path to saved file.
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        
        path = self.save_dir / f"{experiment_name}_hparams.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        return str(path)
