"""
app.py - FastAPI backend for sign language recognition.

Endpoints:
    GET  /health           → Server health check
    GET  /classes          → List available gesture classes
    POST /predict          → Predict gesture from sequence
    POST /predict/batch    → Predict multiple sequences

Usage:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_loader import registry


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sign Language Recognition API",
    description="Real-time sign language gesture prediction using LSTM + MediaPipe keypoints.",
    version="1.0.0",
)

# Allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load model once when the server starts."""
    print("[Startup] Loading model...")
    try:
        registry.load()
        print("[Startup] Model ready.")
    except FileNotFoundError as e:
        print(f"[Startup WARNING] Could not load model: {e}")
        print("[Startup] Server will start but /predict will return errors until model is available.")


@app.on_event("shutdown")
async def shutdown_event():
    print("[Shutdown] Cleaning up.")


# ─── Request / Response Models ────────────────────────────────────────────────

class SequenceInput(BaseModel):
    """
    A single gesture sequence for prediction.

    sequence: 2D list of shape (seq_len, 63).
              Each row contains 63 floats — x, y, z for 21 hand landmarks.
    """
    sequence: List[List[float]] = Field(
        ...,
        description="2D array of shape (seq_len, 63). Each row = one frame of keypoints.",
        example=[[0.0] * 63] * 30,
    )

    @validator("sequence")
    def validate_sequence(cls, v):
        if len(v) == 0:
            raise ValueError("sequence must not be empty")
        if len(v[0]) != 63:
            raise ValueError(f"Each frame must have 63 features, got {len(v[0])}")
        if len(v) < 5:
            raise ValueError(f"Sequence too short ({len(v)} frames). Minimum is 5.")
        return v


class BatchSequenceInput(BaseModel):
    """Multiple sequences for batch prediction."""
    sequences: List[List[List[float]]] = Field(
        ...,
        description="List of sequences. Each sequence is shape (seq_len, 63).",
    )
    confidence_threshold: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to return a prediction (others return null).",
    )


class PredictionResponse(BaseModel):
    label: str
    label_index: int
    confidence: float
    probabilities: dict
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[Optional[PredictionResponse]]
    total_processing_time_ms: float


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health_check():
    """Return server and model status."""
    return {
        "status": "ok",
        "model_loaded": registry.is_loaded,
        "classes": registry.classes if registry.is_loaded else [],
        "device": str(registry.device) if registry.is_loaded else "unknown",
    }


@app.get("/classes", tags=["Meta"])
async def get_classes():
    """Return the list of recognizable gesture classes."""
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")
    return {"classes": registry.classes, "count": len(registry.classes)}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(body: SequenceInput):
    """
    Predict the gesture from a single keypoint sequence.

    Accepts a 2D array of shape (seq_len, 63) and returns:
    - label: Predicted class name
    - label_index: Integer class index
    - confidence: Softmax probability of the predicted class
    - probabilities: Full probability distribution over all classes
    """
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait or check logs.")

    t0 = time.perf_counter()

    try:
        sequence = np.array(body.sequence, dtype=np.float32)  # (seq_len, 63)

        # Pad or truncate to model's expected sequence length
        expected_len = registry.config.get("sequence_length", 30)
        if sequence.shape[0] < expected_len:
            pad = np.zeros((expected_len - sequence.shape[0], 63), dtype=np.float32)
            sequence = np.vstack([sequence, pad])
        elif sequence.shape[0] > expected_len:
            sequence = sequence[:expected_len]

        result = registry.predict(sequence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    processing_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        label=result["label"],
        label_index=result["label_index"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        processing_time_ms=round(processing_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(body: BatchSequenceInput):
    """
    Predict gestures for a batch of sequences.

    Returns a list of predictions (or null for low-confidence results).
    """
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(body.sequences) > 64:
        raise HTTPException(status_code=400, detail="Maximum 64 sequences per batch.")

    t0 = time.perf_counter()
    predictions = []
    expected_len = registry.config.get("sequence_length", 30)

    for raw_seq in body.sequences:
        try:
            sequence = np.array(raw_seq, dtype=np.float32)

            # Validate features
            if sequence.ndim != 2 or sequence.shape[1] != 63:
                predictions.append(None)
                continue

            # Pad/truncate
            if sequence.shape[0] < expected_len:
                pad = np.zeros((expected_len - sequence.shape[0], 63), dtype=np.float32)
                sequence = np.vstack([sequence, pad])
            elif sequence.shape[0] > expected_len:
                sequence = sequence[:expected_len]

            t_seq = time.perf_counter()
            result = registry.predict(sequence)
            seq_ms = (time.perf_counter() - t_seq) * 1000

            if result["confidence"] >= body.confidence_threshold:
                predictions.append(
                    PredictionResponse(
                        label=result["label"],
                        label_index=result["label_index"],
                        confidence=result["confidence"],
                        probabilities=result["probabilities"],
                        processing_time_ms=round(seq_ms, 2),
                    )
                )
            else:
                predictions.append(None)

        except Exception:
            predictions.append(None)

    total_ms = (time.perf_counter() - t0) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_processing_time_ms=round(total_ms, 2),
    )


# ─── Exception Handler ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
