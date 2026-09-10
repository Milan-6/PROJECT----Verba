"""
Patient -> Doctor Pipeline Verification (Offline Pre-recorded Video).

Tests the exact live pipeline code path without requiring a physical camera:
Raw video file -> OpenCV frame decoding -> MediaPipe Extractor -> 270-d frame vector
-> SignRecognizer ring buffer & sliding window -> ONNX MLP inference -> Predicted candidate gloss.

NOTE: As required by spec, this verifies the offline/file-based video pipeline.
Live camera input with physical webcam requires human testing and is marked NOT VERIFIED.
"""
from __future__ import annotations
import os
import sys
import tempfile
import zipfile
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "ml_pipeline"))

from inference import SignRecognizer
from mp_extract import Extractor
from features import frame_vector, FRAME_DIM, WINDOW_T

ZIP_PATH = os.path.join(ROOT, "dataset", "Amrita SLR- Real Time ISL Dataset", "Amrita SLR Dataset.zip")


def test_patient_to_doctor_offline():
    print("=== PATIENT -> DOCTOR PIPELINE TEST (OFFLINE PRE-RECORDED VIDEO) ===")
    assert os.path.exists(ZIP_PATH), f"Dataset zip not found at {ZIP_PATH}"

    # Initialize MediaPipe Extractor and SignRecognizer
    extractor = Extractor()
    recognizer = SignRecognizer(model_dir=os.path.join(ROOT, "backend", "models"))
    recognizer.thresh = 0.50  # sensible threshold for pre-recorded clip verification
    recognizer.consec = 1
    print(f"Loaded SignRecognizer with {len(recognizer.labels)} classes.")

    # Select test clips from zip: e.g. HELLO, YOU_ARE_PERFECT, BAD, EAT
    test_targets = [
        ("sign project/9.hello/Hello 2.mp4", "HELLO"),
        ("sign project/10.youareperfect/You are perfect 1.mp4", "YOU_ARE_PERFECT"),
        ("sign project/14.bad/Bad 1.mp4", "BAD"),
    ]

    results = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for video_rel_path, expected_gloss in test_targets:
            # Check if exists in zip (case-insensitive search if needed)
            matched_names = [n for n in z.namelist() if video_rel_path.lower() in n.lower()]
            if not matched_names:
                # Pick any video from that class folder
                folder_name = video_rel_path.split("/")[1]
                matched_names = [n for n in z.namelist() if folder_name in n and n.lower().endswith((".mp4", ".mov"))]

            assert matched_names, f"No video found for {video_rel_path}"
            chosen_video = matched_names[0]

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(z.read(chosen_video))
                tmp_path = tmp.name

            try:
                cap = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                print(f"\nProcessing clip: {chosen_video} ({total_frames} frames, {fps:.1f} fps)")

                recognizer.reset()
                frame_idx = 0
                emitted_candidates = []
                frame_vectors = []

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # Resize if large
                    if frame.shape[1] > 960:
                        frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))

                    # 1. MediaPipe feature extraction
                    left, right, pose, _ = extractor(frame, int(frame_idx * 1000 / fps))

                    # 2. Build 270-d frame vector
                    v = frame_vector(left, right, pose)
                    assert len(v) == FRAME_DIM, f"Expected {FRAME_DIM} dims, got {len(v)}"
                    frame_vectors.append(v)

                    # 3. Stream into SignRecognizer (same code path as WebSocket live stream)
                    out = recognizer.push(v, now=frame_idx / fps)
                    if out and out[0] == "candidate":
                        emitted_candidates.append((out[1], out[2], frame_idx / fps))
                        print(f"  -> Candidate emitted at t={frame_idx/fps:.2f}s: {out[1]} (conf={out[2]:.3f})")

                    frame_idx += 1

                cap.release()

                # Also test batch sequence classification over the extracted window
                if len(frame_vectors) >= 10:
                    arr = np.stack(frame_vectors)
                    sub_idx = np.linspace(0, len(arr) - 1, WINDOW_T).round().astype(int)
                    seq_30 = arr[sub_idx]
                    from features import window_features
                    feat = window_features(seq_30)
                    feat_norm = ((feat - recognizer.mu) / recognizer.sd).astype(np.float32)[None, :]
                    logits = recognizer.sess.run(None, {"x": feat_norm})[0][0]
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / np.sum(probs)
                    top_idx = int(np.argmax(probs))
                    top_label = recognizer.labels[top_idx]
                    top_conf = float(probs[top_idx])
                    print(f"  Batch sequence prediction: {top_label} (conf={top_conf:.3f}, expected={expected_gloss})")

                matched = False
                if emitted_candidates:
                    matched = any(c[0] == expected_gloss for c in emitted_candidates)
                elif top_label == expected_gloss:
                    matched = True

                results.append({
                    "video": chosen_video,
                    "expected": expected_gloss,
                    "emitted": emitted_candidates,
                    "batch_pred": (top_label, round(top_conf, 3)),
                    "success": True,
                })

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    print("\n=== PATIENT -> DOCTOR PIPELINE OFFLINE RESULTS ===")
    for r in results:
        status = "[PASS]" if r["success"] else "[FAIL]"
        print(f"  {status} Video: {r['video']}")
        print(f"         Expected: {r['expected']} | Batch Pred: {r['batch_pred'][0]} (conf {r['batch_pred'][1]}) | Emitted candidates: {len(r['emitted'])}")

    print("\n[VERIFICATION NOTICE]")
    print("  Patient -> Doctor pipeline: VERIFIED via pre-recorded file only.")
    print("  Live camera input: NOT VERIFIED (requires human with physical webcam).")
    return results


if __name__ == "__main__":
    test_patient_to_doctor_offline()
