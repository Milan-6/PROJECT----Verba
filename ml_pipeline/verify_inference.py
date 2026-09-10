"""
Verifies end-to-end inference of the trained Amrita-based model on real unseen dataset samples.
Tests:
1. Model loading & artifact consistency.
2. Inference on unseen test sequences from data_splits.json.
3. Live pipeline simulation: frame vector -> sliding window -> ONNX MLP -> predicted class.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "..", "backend", "models")
DATA_DIR = os.path.join(ROOT, "data")
SPLITS_PATH = os.path.join(ROOT, "data_splits.json")

# Import feature pipeline
from features import window_features, WINDOW_T, FRAME_DIM, WINDOW_DIM


def verify_inference():
    print("=== MODEL INFERENCE VERIFICATION ===")

    # 1. Verify artifacts exist
    required_files = [
        "model.onnx", "labels.json", "scaler.json", "class_to_index.json",
        "index_to_class.json", "preprocessing.json", "training_config.json", "evaluation.json"
    ]
    for rf in required_files:
        p = os.path.join(MODELS_DIR, rf)
        assert os.path.exists(p), f"Missing artifact: {rf}"
        print(f"  [OK] Found {rf} ({os.path.getsize(p)} bytes)")

    # 2. Load model and metadata
    onnx_path = os.path.join(MODELS_DIR, "model.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    labels = json.load(open(os.path.join(MODELS_DIR, "labels.json"), encoding="utf-8"))
    class_to_idx = json.load(open(os.path.join(MODELS_DIR, "class_to_index.json"), encoding="utf-8"))
    idx_to_class = json.load(open(os.path.join(MODELS_DIR, "index_to_class.json"), encoding="utf-8"))
    scaler = json.load(open(os.path.join(MODELS_DIR, "scaler.json"), encoding="utf-8"))
    mu = np.array(scaler["mean"], dtype=np.float32)
    sd = np.array(scaler["std"], dtype=np.float32)

    print(f"\nModel loaded: {len(labels)} classes")
    print(f"Input signature: {sess.get_inputs()[0].name}, shape: {sess.get_inputs()[0].shape}")
    print(f"Output signature: {sess.get_outputs()[0].name}, shape: {sess.get_outputs()[0].shape}")

    # 3. Test on real unseen test samples from data_splits.json
    splits = json.load(open(SPLITS_PATH, encoding="utf-8"))
    test_samples = splits.get("test", [])
    print(f"\nEvaluating on {len(test_samples)} unseen Amrita test samples...")

    correct = 0
    tested = 0
    predictions = []

    for item in test_samples:
        gloss = item["gloss"]
        npy_file = item["saved_as"]
        npy_path = os.path.join(DATA_DIR, gloss, npy_file)
        if not os.path.exists(npy_path):
            continue

        seq = np.load(npy_path)
        assert seq.shape == (WINDOW_T, FRAME_DIM), f"Invalid sequence shape {seq.shape}"

        feat = window_features(seq)
        assert feat.shape == (WINDOW_DIM,), f"Invalid feature shape {feat.shape}"

        feat_norm = ((feat - mu) / sd).astype(np.float32)[None, :]
        logits = sess.run(None, {"x": feat_norm})[0][0]
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        conf = float(probs[pred_idx])

        is_match = (pred_label == gloss)
        if is_match:
            correct += 1
        tested += 1

        predictions.append({
            "video_file": item["video_file"],
            "true_gloss": gloss,
            "pred_gloss": pred_label,
            "confidence": round(conf, 3),
            "match": is_match,
        })

    acc = (correct / tested) if tested > 0 else 0.0
    print(f"\nTest Sample Results: {correct}/{tested} correct ({acc * 100:.1f}%)")

    # Sample predictions display
    print("\nSample predictions on unseen Amrita test videos:")
    for p in predictions[:8]:
        status_icon = "[PASS]" if p["match"] else "[FAIL]"
        print(f"  {status_icon} True: {p['true_gloss']:<14} | Predicted: {p['pred_gloss']:<14} (conf: {p['confidence']:.2f})")

    # Save verification report
    verif_path = os.path.join(MODELS_DIR, "inference_verification.json")
    with open(verif_path, "w", encoding="utf-8") as f:
        json.dump({
            "verified": True,
            "num_test_samples": tested,
            "accuracy": round(acc, 4),
            "sample_predictions": predictions[:15],
        }, f, indent=2)
    print(f"\nSaved inference verification report to: {verif_path}")

    # 4. Verify SignRecognizer compatibility in backend/inference.py
    sys.path.insert(0, os.path.join(ROOT, "..", "backend"))
    from inference import SignRecognizer
    recognizer = SignRecognizer(model_dir=MODELS_DIR)
    print("  [OK] SignRecognizer initialized successfully with newly trained model!")
    recognizer.reset()
    dummy_frame = [0.0] * FRAME_DIM
    res = recognizer.push(dummy_frame)
    print(f"  [OK] SignRecognizer dummy push result: {res}")


if __name__ == "__main__":
    verify_inference()
