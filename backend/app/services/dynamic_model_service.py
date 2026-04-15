from __future__ import annotations

from collections import defaultdict, deque
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from app.services.model_service import MODEL_DIR


BASE_DIR = Path(__file__).resolve().parents[3]
DYNAMIC_MODEL_PATH = BASE_DIR / "Dynamic_M" / "models" / "best_model.pth"
LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
DYNAMIC_LABELS_PATH = BASE_DIR / "Dynamic_M" / "labels.json"
DYNAMIC_CONFIG_PATH = BASE_DIR / "Dynamic_M" / "models" / "training_config.json"

# Recovered from Dynamic_M compiled scripts; trimmed or expanded as needed to
# match the trained classifier output width in best_model.pth.
_DEFAULT_DYNAMIC_CLASSES = ["hello", "bye", "yes", "no", "please", "sorry", "ily", "thank_you", "help"]


class DynamicInvalidImageError(ValueError):
    pass


class DynamicModelInitializationError(RuntimeError):
    pass


class _SignLanguageLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_timestep = out[:, -1, :]
        return self.classifier(last_timestep)


class _DynamicInferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _SignLanguageLSTM | None = None
        self.detector: Any = None
        self.classes: list[str] = []
        self.sequence_length = 30
        self.confidence_threshold = 0.80
        self._checkpoint_source: str | None = None
        self._is_trained_weights = False
        self._sequence_buffers: dict[str, deque[np.ndarray]] = {}
        self._prediction_buffers: dict[str, deque[tuple[int, float]]] = {}
        self._prediction_window = 5
        self._model_config: dict[str, Any] = {}
        self._label_map: dict[str, int] = {}

    def initialize(self) -> None:
        if self.model is not None and self.detector is not None:
            return

        if not DYNAMIC_MODEL_PATH.exists():
            raise DynamicModelInitializationError(f"Dynamic model checkpoint not found: {DYNAMIC_MODEL_PATH}")

        try:
            self._model_config = self._load_model_config()
            self.classes, self._label_map = self._load_labels()
            self.sequence_length = int(self._model_config.get("sequence_length", self.sequence_length))

            state_dict = self._load_state_dict(DYNAMIC_MODEL_PATH)
            self.model = self._build_model_from_state_dict(state_dict)
            self.model.eval()
            self.detector = self._create_detector()
            self._checkpoint_source = str(DYNAMIC_MODEL_PATH)
            self._is_trained_weights = True
        except Exception as exc:
            self.shutdown()
            raise DynamicModelInitializationError(str(exc)) from exc

    def shutdown(self) -> None:
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
        self.detector = None
        self.model = None
        self.classes = []
        self._checkpoint_source = None
        self._is_trained_weights = False
        self._sequence_buffers.clear()
        self._prediction_buffers.clear()
        self._model_config = {}
        self._label_map = {}

    def status(self) -> dict[str, object]:
        source = Path(self._checkpoint_source).name if self._checkpoint_source else None
        return {
            "model_loaded": self.model is not None and self.detector is not None,
            "is_trained_weights": self._is_trained_weights,
            "checkpoint_source": source,
            "num_classes": len(self.classes),
            "classes": list(self.classes),
            "device": str(self.device),
        }

    def reset_session(self, session_id: str) -> None:
        self._sequence_buffers.pop(session_id, None)
        self._prediction_buffers.pop(session_id, None)

    def predict(self, image_bytes: bytes, session_id: str) -> dict[str, object]:
        if self.model is None or self.detector is None:
            raise DynamicModelInitializationError("Dynamic model service is not initialized.")

        image_bgr = self._decode_image(image_bytes)
        keypoints, hand_detected = self._extract_landmarks(image_bgr)
        keypoints = self._normalize_keypoints(keypoints)

        seq_buffer = self._sequence_buffers.setdefault(session_id, deque(maxlen=self.sequence_length))
        pred_buffer = self._prediction_buffers.setdefault(session_id, deque(maxlen=self._prediction_window))
        seq_buffer.append(keypoints)
        frames_collected = len(seq_buffer)
        if frames_collected < self.sequence_length:
            return {
                "ready": False,
                "frames_collected": frames_collected,
                "frames_required": self.sequence_length,
                "prediction": None,
                "confidence": None,
                "hand_detected": hand_detected,
            }

        with torch.no_grad():
            seq = np.stack(seq_buffer, axis=0)
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)

        pred_idx = int(idx.item())
        confidence = float(conf.item())
        pred_buffer.append((pred_idx, confidence))
        stable_idx, smoothed_conf = self._smooth_prediction(pred_buffer)

        # Keep standalone-like smoothing, but never drop prediction entirely
        # once a full sequence is available; fallback to raw top-1.
        if stable_idx is None:
            stable_idx = pred_idx
            smoothed_conf = confidence

        prediction = self.classes[stable_idx] if stable_idx is not None and stable_idx < len(self.classes) else None

        return {
            "ready": True,
            "frames_collected": self.sequence_length,
            "frames_required": self.sequence_length,
            "prediction": prediction,
            "confidence": smoothed_conf,
            "hand_detected": hand_detected,
        }

    def _smooth_prediction(self, buffer: deque[tuple[int, float]]) -> tuple[int | None, float]:
        if not buffer:
            return None, 0.0

        label_counts: dict[int, int] = defaultdict(int)
        confidences: list[float] = []
        for label_idx, conf in buffer:
            label_counts[label_idx] += 1
            confidences.append(conf)

        most_common_idx = max(label_counts, key=label_counts.get)
        count = label_counts[most_common_idx]
        avg_confidence = float(np.mean(confidences))
        majority = (self._prediction_window // 2) + 1
        if count >= majority and avg_confidence >= self.confidence_threshold:
            return most_common_idx, avg_confidence
        return None, 0.0

    def _extract_landmarks(self, image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        image_rgb = image_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        if not result.hand_landmarks:
            return np.zeros(63, dtype=np.float32), False

        lm = result.hand_landmarks[0]
        keypoints = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
        return keypoints, True

    @staticmethod
    def _normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
        # Mirror Dynamic_M/utils.py normalize_keypoints.
        if np.all(keypoints == 0):
            return keypoints

        kp = keypoints.reshape(21, 3)
        wrist = kp[0].copy()
        kp -= wrist
        max_val = np.max(np.abs(kp))
        if max_val > 0:
            kp /= max_val
        return kp.flatten()

    def _create_detector(self) -> Any:
        if not LANDMARKER_PATH.exists():
            raise DynamicModelInitializationError(f"Hand Landmarker asset not found: {LANDMARKER_PATH}")

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                rgb_array = np.array(image, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise DynamicInvalidImageError("Invalid image file.") from exc
        return rgb_array[:, :, ::-1].copy()

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            for key in ("model_state", "state_dict", "model_state_dict", "net", "weights"):
                nested = loaded.get(key)
                if isinstance(nested, dict) and nested:
                    tensors = {k: v for k, v in nested.items() if isinstance(v, torch.Tensor)}
                    if tensors:
                        return tensors
            tensors = {k: v for k, v in loaded.items() if isinstance(v, torch.Tensor)}
            if tensors:
                return tensors
        raise DynamicModelInitializationError("Invalid dynamic checkpoint format.")

    def _load_model_config(self) -> dict[str, Any]:
        if DYNAMIC_CONFIG_PATH.exists():
            with DYNAMIC_CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        return {}

    def _load_labels(self) -> tuple[list[str], dict[str, int]]:
        if DYNAMIC_LABELS_PATH.exists():
            with DYNAMIC_LABELS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                classes = data.get("classes")
                label_map = data.get("label_map")
                if isinstance(classes, list) and isinstance(label_map, dict):
                    classes_str = [str(c) for c in classes]
                    label_map_int = {str(k): int(v) for k, v in label_map.items()}
                    return classes_str, label_map_int

        classes = self._model_config.get("classes")
        if isinstance(classes, list) and classes:
            classes_str = [str(c) for c in classes]
            label_map = {cls: idx for idx, cls in enumerate(classes_str)}
            return classes_str, label_map

        return self._resolve_classes(int(self._model_config.get("num_classes", 0) or 0)), {}

    def _build_model_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> _SignLanguageLSTM:
        input_size = int(self._model_config.get("input_size", state_dict["lstm.weight_ih_l0"].shape[1]))
        hidden_size = int(self._model_config.get("hidden_size", state_dict["lstm.weight_hh_l0"].shape[1]))
        num_layers = int(self._model_config.get("num_layers", len([k for k in state_dict if k.startswith("lstm.weight_ih_l")])))
        num_classes = int(self._model_config.get("num_classes", state_dict["classifier.4.weight"].shape[0]))
        # Dynamic_M inference_utils.py passes dropout=0.0 when creating inference model.
        dropout = 0.0

        if not self.classes:
            self.classes = self._resolve_classes(num_classes)
        model = _SignLanguageLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)
        model.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def _resolve_classes(num_classes: int) -> list[str]:
        if num_classes <= len(_DEFAULT_DYNAMIC_CLASSES):
            return _DEFAULT_DYNAMIC_CLASSES[:num_classes]
        extras = [f"class_{i}" for i in range(len(_DEFAULT_DYNAMIC_CLASSES), num_classes)]
        return _DEFAULT_DYNAMIC_CLASSES + extras


dynamic_inference_service = _DynamicInferenceService()


def initialize_dynamic_model_service() -> None:
    dynamic_inference_service.initialize()


def shutdown_dynamic_model_service() -> None:
    dynamic_inference_service.shutdown()


def get_dynamic_model_status() -> dict[str, object]:
    return dynamic_inference_service.status()


def predict_dynamic_from_image(image_bytes: bytes, session_id: str = "default") -> dict[str, object]:
    return dynamic_inference_service.predict(image_bytes, session_id=session_id)


def reset_dynamic_session(session_id: str) -> None:
    dynamic_inference_service.reset_session(session_id)
