"""
realtime.py - Real-time sign language gesture recognition using webcam.

Usage:
    python realtime.py [--model ../models/best_model.pth]
                       [--config ../models/training_config.json]
                       [--labels ../labels.json]
                       [--confidence 0.8]
                       [--smooth 5]
                       [--camera 0]

Controls:
    Press 'q' to quit.
    Press 'r' to reset the sequence buffer.
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_mediapipe_hands,
    extract_keypoints,
    normalize_keypoints,
    draw_landmarks,
    put_text_with_background,
)
from inference_utils import (
    load_labels,
    load_model,
    PredictionSmoother,
    FPSCounter,
    SequenceBuffer,
    predict_sequence,
)


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition")
    parser.add_argument("--model",      type=str, default="models/best_model.pth")
    parser.add_argument("--config",     type=str, default="models/training_config.json")
    parser.add_argument("--labels",     type=str, default="labels.json")
    parser.add_argument("--confidence", type=float, default=0.8, help="Min confidence to display")
    parser.add_argument("--smooth",     type=int,   default=5,   help="Prediction smoothing window")
    parser.add_argument("--camera",     type=int,   default=0,   help="Webcam index")
    parser.add_argument("--seq_len",    type=int,   default=30,  help="Frames per sequence")
    return parser.parse_args()


# ─── Overlay Drawing ──────────────────────────────────────────────────────────

def draw_sequence_progress(frame: np.ndarray, current: int, total: int):
    """Draw a frame-progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_y = h - 15
    bar_x0, bar_x1 = 10, w - 10
    bar_w = bar_x1 - bar_x0

    # Background
    cv2.rectangle(frame, (bar_x0, bar_y - 8), (bar_x1, bar_y + 8), (50, 50, 50), cv2.FILLED)
    # Filled portion
    filled_x = bar_x0 + int(bar_w * current / total)
    cv2.rectangle(frame, (bar_x0, bar_y - 8), (filled_x, bar_y + 8), (0, 200, 100), cv2.FILLED)
    # Text
    cv2.putText(
        frame, f"{current}/{total}", (bar_x0 + bar_w // 2 - 20, bar_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_probability_bars(
    frame: np.ndarray,
    probabilities: np.ndarray,
    classes: list,
    x_offset: int = None,
):
    """Draw a small probability bar chart on the right side of the frame."""
    h, w = frame.shape[:2]
    if x_offset is None:
        x_offset = w - 170

    bar_h = 15
    bar_max_w = 150
    margin = 5
    y_start = 50

    for i, (cls, prob) in enumerate(zip(classes, probabilities)):
        y = y_start + i * (bar_h + margin)
        bar_w = int(bar_max_w * prob)

        # Background
        cv2.rectangle(frame, (x_offset, y), (x_offset + bar_max_w, y + bar_h), (40, 40, 40), cv2.FILLED)
        # Probability bar (green if top, blue otherwise)
        color = (0, 220, 0) if i == int(np.argmax(probabilities)) else (100, 150, 255)
        cv2.rectangle(frame, (x_offset, y), (x_offset + bar_w, y + bar_h), color, cv2.FILLED)
        # Label + value
        cv2.putText(
            frame, f"{cls[:8]}: {prob:.2f}",
            (x_offset - 95, y + bar_h - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
        )


# ─── Main Inference Loop ──────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Setup ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model + labels
    classes, label_map = load_labels(args.labels)
    model = load_model(args.model, args.config, device)

    # Inference helpers
    seq_buffer = SequenceBuffer(sequence_length=args.seq_len, feature_size=63)
    smoother = PredictionSmoother(window_size=args.smooth, confidence_threshold=args.confidence)
    fps_counter = FPSCounter(window=30)

    # MediaPipe
    hands = get_mediapipe_hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State
    current_prediction = None
    current_confidence = 0.0
    last_probabilities = np.zeros(len(classes))

    print(f"\n[Inference] Running. Press 'q' to quit, 'r' to reset buffer.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            fps_counter.tick()

            # ── MediaPipe processing ─────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # Draw hand landmarks
            frame = draw_landmarks(frame, results)

            # Extract + normalize keypoints
            kp = extract_keypoints(results)
            kp = normalize_keypoints(kp)

            # Add to rolling buffer
            seq_buffer.add_frame(kp)

            # ── Inference when buffer is full ────────────────────────────────
            if seq_buffer.is_full:
                sequence = seq_buffer.get_sequence()
                pred_idx, confidence, probabilities = predict_sequence(model, sequence, device)
                last_probabilities = probabilities

                # Update smoother
                smoother.update(pred_idx, confidence)
                stable_label, stable_conf = smoother.get_stable_prediction()

                if stable_label is not None:
                    current_prediction = classes[stable_label]
                    current_confidence = stable_conf
                else:
                    # Keep last prediction if not confident enough
                    pass

            # ── Overlay UI ───────────────────────────────────────────────────
            h, w = frame.shape[:2]

            # FPS
            put_text_with_background(
                frame, f"FPS: {fps_counter.fps:.1f}",
                (10, 30), font_scale=0.7,
                color=(200, 200, 200), bg_color=(0, 0, 0),
            )

            # Main prediction label
            if current_prediction:
                label_text = f"{current_prediction.upper()}  {current_confidence*100:.1f}%"
                put_text_with_background(
                    frame, label_text,
                    (10, h // 2),
                    font_scale=1.6,
                    color=(0, 255, 100),
                    bg_color=(0, 0, 0),
                    thickness=3,
                )
            else:
                put_text_with_background(
                    frame, "Waiting for gesture...",
                    (10, h // 2),
                    font_scale=1.0,
                    color=(150, 150, 150),
                    bg_color=(0, 0, 0),
                )

            # Sequence progress bar
            draw_sequence_progress(frame, len(seq_buffer._buffer), args.seq_len)

            # Probability bars (right side)
            draw_probability_bars(frame, last_probabilities, classes)

            # Instructions
            put_text_with_background(
                frame, "q: quit | r: reset",
                (10, h - 40),
                font_scale=0.55,
                color=(180, 180, 180),
                bg_color=(0, 0, 0),
            )

            cv2.imshow("Sign Language Recognition — Real-time", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                seq_buffer.reset()
                smoother.reset()
                current_prediction = None
                print("[Reset] Buffer cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("[Exit] Webcam released.")


if __name__ == "__main__":
    main()
