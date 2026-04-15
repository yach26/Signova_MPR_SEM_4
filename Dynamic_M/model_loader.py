"""
model_loader.py - Singleton model loader for the FastAPI backend.

The model is loaded once at application startup and reused for all requests,
avoiding repeated disk I/O and initialization overhead.
"""

import os
import sys
import json
from typing import List, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Default Paths ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pth")
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "models", "training_config.json")
DEFAULT_LABELS_PATH = os.path.join(BASE_DIR, "labels.json")


# ─── Model Registry (Singleton) ───────────────────────────────────────────────

class ModelRegistry:
    """
    Holds the loaded model, class names, and device.
    Implemented as a module-level singleton so FastAPI only loads once.
    """

    def __init__(self):
        self.model = None
        self.classes: List[str] = []
        self.device: torch.device = torch.device("cpu")
        self.config: dict = {}
        self._loaded = False

    def load(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        labels_path: str = DEFAULT_LABELS_PATH,
    ):
        """
        Load the model and metadata from disk.

        Args:
            model_path: Path to .pth checkpoint.
            config_path: Path to training_config.json.
            labels_path: Path to labels.json.
        """
        if self._loaded:
            print("[ModelRegistry] Already loaded — skipping.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelRegistry] Device: {self.device}")

        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load classes
        with open(labels_path, "r") as f:
            label_data = json.load(f)
        self.classes = label_data["classes"]

        # Load model
        from model import SignLanguageLSTM
        self.model = SignLanguageLSTM(
            input_size=self.config.get("input_size", 63),
            hidden_size=self.config.get("hidden_size", 128),
            num_layers=self.config.get("num_layers", 2),
            num_classes=self.config["num_classes"],
            dropout=0.0,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self._loaded = True
        print(f"[ModelRegistry] Model loaded. Classes: {self.classes}")

    @torch.no_grad()
    def predict(self, sequence: np.ndarray):
        """
        Run inference on a sequence.

        Args:
            sequence: np.ndarray of shape (seq_len, 63).

        Returns:
            dict with keys: label, label_index, confidence, probabilities.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call registry.load() first.")

        x = torch.tensor(sequence[np.newaxis], dtype=torch.float32, device=self.device)
        probs = torch.softmax(self.model(x), dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "label": self.classes[pred_idx],
            "label_index": pred_idx,
            "confidence": confidence,
            "probabilities": {cls: float(p) for cls, p in zip(self.classes, probs)},
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Module-level singleton — import this in app.py
registry = ModelRegistry()
