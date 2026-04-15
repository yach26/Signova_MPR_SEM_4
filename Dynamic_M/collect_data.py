"""
collect_data.py - Data collection script for sign language recognition.

Usage:
    python collect_data.py

Controls:
    Press 'q' to quit at any time.
    Press 's' to skip current class.
    The script automatically cycles through all gesture classes.
"""

import os
import sys
import time
import cv2
import numpy as np

# Allow imports from local directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_mediapipe_hands,
    extract_keypoints,
    normalize_keypoints,
    draw_landmarks,
    put_text_with_background,
)

# ─── Configuration ───────────────────────────────────────────────────────────

# Gesture classes to collect data for
CLASSES = ["hello", "bye", "yes", "no", "please", "sorry", "ily", "thank_you"]

# Number of sequences (videos) to collect per class
NUM_SEQUENCES = 45

# Number of frames per sequence
SEQUENCE_LENGTH = 45

# Path to save dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# Countdown seconds before each recording starts
COUNTDOWN_SECONDS = 5

# Camera index (0 = default webcam)
CAMERA_INDEX = 0


# ─── Setup ───────────────────────────────────────────────────────────────────

def create_dataset_folders():
    """Create dataset directory structure for all classes."""
    for cls in CLASSES:
        class_path = os.path.join(DATASET_PATH, cls)
        os.makedirs(class_path, exist_ok=True)
        print(f"  [OK] {class_path}")


def get_next_sequence_index(class_name: str) -> int:
    """
    Find the next available sequence index for a class.
    This allows resuming collection without overwriting existing data.
    """
    class_path = os.path.join(DATASET_PATH, class_name)
    existing = [
        f for f in os.listdir(class_path) if f.endswith(".npy")
    ]
    return len(existing)


# ─── Recording ───────────────────────────────────────────────────────────────

def show_countdown(frame: np.ndarray, seconds: int, class_name: str) -> np.ndarray:
    """Overlay countdown text on frame."""
    frame = frame.copy()
    put_text_with_background(
        frame,
        f"Get ready: {class_name.upper()}",
        (10, 40),
        font_scale=1.2,
        color=(0, 255, 255),
        bg_color=(0, 0, 0),
        thickness=2,
    )
    put_text_with_background(
        frame,
        f"Starting in {seconds}...",
        (10, 90),
        font_scale=1.0,
        color=(255, 255, 0),
        bg_color=(0, 0, 0),
        thickness=2,
    )
    return frame


def collect_sequence(
    cap: cv2.VideoCapture,
    hands,
    class_name: str,
    sequence_idx: int,
) -> bool:
    """
    Collect a single sequence of SEQUENCE_LENGTH frames.

    Args:
        cap: OpenCV VideoCapture object.
        hands: MediaPipe Hands instance.
        class_name: Name of the gesture class.
        sequence_idx: Index of this sequence within the class.

    Returns:
        True if sequence was saved, False if user quit.
    """
    keypoints_buffer = []

    # ── Countdown phase ──────────────────────────────────────────────────────
    countdown_start = time.time()
    while time.time() - countdown_start < COUNTDOWN_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            return False

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame = draw_landmarks(frame, results)

        remaining = COUNTDOWN_SECONDS - int(time.time() - countdown_start)
        frame = show_countdown(frame, remaining, class_name)

        # Progress bar for sequences
        progress_text = f"Seq {sequence_idx + 1}/{NUM_SEQUENCES} | Class: {class_name}"
        put_text_with_background(
            frame, progress_text, (10, frame.shape[0] - 20),
            font_scale=0.7, color=(200, 200, 200), bg_color=(0, 0, 0),
        )

        cv2.imshow("Sign Language Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("s"):
            return True  # Skip

    # ── Recording phase ──────────────────────────────────────────────────────
    for frame_idx in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame during recording.")
            return False

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame = draw_landmarks(frame, results)

        # Extract and normalize keypoints
        kp = extract_keypoints(results)
        kp = normalize_keypoints(kp)
        keypoints_buffer.append(kp)

        # Visual feedback during recording
        put_text_with_background(
            frame,
            f"RECORDING: {class_name.upper()}",
            (10, 40),
            font_scale=1.2,
            color=(0, 0, 255),
            bg_color=(0, 0, 0),
            thickness=2,
        )

        # Progress bar (frame count)
        bar_width = 300
        filled = int(bar_width * frame_idx / SEQUENCE_LENGTH)
        cv2.rectangle(frame, (10, 65), (10 + bar_width, 85), (50, 50, 50), cv2.FILLED)
        cv2.rectangle(frame, (10, 65), (10 + filled, 85), (0, 200, 0), cv2.FILLED)
        put_text_with_background(
            frame,
            f"Frame {frame_idx + 1}/{SEQUENCE_LENGTH}",
            (10, 110),
            font_scale=0.7,
            color=(200, 200, 200),
            bg_color=(0, 0, 0),
        )
        put_text_with_background(
            frame,
            f"Seq {sequence_idx + 1}/{NUM_SEQUENCES}",
            (10, frame.shape[0] - 20),
            font_scale=0.7,
            color=(200, 200, 200),
            bg_color=(0, 0, 0),
        )

        cv2.imshow("Sign Language Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

    # ── Save sequence ────────────────────────────────────────────────────────
    sequence_array = np.array(keypoints_buffer, dtype=np.float32)  # (30, 63)
    save_path = os.path.join(DATASET_PATH, class_name, f"{sequence_idx}.npy")
    np.save(save_path, sequence_array)
    print(f"  [SAVED] {save_path} | shape={sequence_array.shape}")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Sign Language Data Collection")
    print("=" * 60)
    print(f"  Classes     : {CLASSES}")
    print(f"  Sequences   : {NUM_SEQUENCES} per class")
    print(f"  Seq length  : {SEQUENCE_LENGTH} frames")
    print(f"  Dataset path: {DATASET_PATH}")
    print("=" * 60 + "\n")

    # Create folders
    print("[1] Creating dataset folders...")
    create_dataset_folders()

    # Open webcam
    print(f"\n[2] Opening camera (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Exiting.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize MediaPipe
    hands = get_mediapipe_hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    print("[3] Starting data collection. Press 'q' to quit, 's' to skip class.\n")

    try:
        for class_name in CLASSES:
            start_seq = get_next_sequence_index(class_name)
            if start_seq >= NUM_SEQUENCES:
                print(f"[SKIP] {class_name} already has {NUM_SEQUENCES} sequences.")
                continue

            print(f"\n{'-' * 40}")
            print(f"  Collecting: {class_name.upper()}")
            print(f"  Resuming from sequence {start_seq}")
            print(f"{'-' * 40}")

            # ── Wait for user to start this class ────────────────────────────
            print(f"[WAIT] Waiting for user to start class: {class_name}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                
                # Show landmarks while waiting so user can position themselves
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                frame = draw_landmarks(frame, results)

                # Show instructions on overlay
                h, w = frame.shape[:2]
                put_text_with_background(
                    frame, f"NEXT WORD: {class_name.upper()}", (w // 4, h // 2 - 40),
                    font_scale=1.2, color=(0, 255, 0), thickness=3
                )
                put_text_with_background(
                    frame, "Press 'c' to CONTINUE / START", (w // 4, h // 2 + 20),
                    font_scale=0.8, color=(255, 255, 255), thickness=2
                )
                put_text_with_background(
                    frame, "Press 'q' to QUIT", (w // 4, h // 2 + 55),
                    font_scale=0.6, color=(200, 200, 200)
                )

                cv2.imshow("Sign Language Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    break
                if key == ord("q"):
                    print("\n[EXIT] User quit.")
                    return

            for seq_idx in range(start_seq, NUM_SEQUENCES):
                success = collect_sequence(cap, hands, class_name, seq_idx)
                if not success:
                    print("\n[EXIT] User quit or error occurred.")
                    return

        print("\n[DONE] Data collection complete!")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
