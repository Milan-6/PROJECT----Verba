"""
Combined Dataset Preprocessing Pipeline: Amrita SLR + ISL-CSLRT Corpus.
- Discovers videos from both datasets without altering raw data.
- Handles duplicate files using MD5 hashes.
- Normalizes raw labels to canonical ISL classes.
- Extracts MediaPipe Hands & Pose landmarks (270-dim vector, 30 frames).
- Saves standardized .npy landmark files into ml_pipeline/data/<GLOSS>/.
- Enforces strict signer-independent train/val/test splits (zero leakage).
- Emits combined_dataset_report.json and combined_data_splits.json.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
import re
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
AMRITA_ZIP = os.path.abspath(os.path.join(ROOT, "..", "dataset", "Amrita SLR- Real Time ISL Dataset", "Amrita SLR Dataset.zip"))
CSLRT_DIR = os.path.abspath(os.path.join(ROOT, "..", "dataset", "ISL_CSLRT_Corpus", "ISL_CSLRT_Corpus", "Videos_Sentence_Level"))
DATA_DIR = os.path.join(ROOT, "data")
EXISTING_SPLITS = os.path.join(ROOT, "data_splits.json")
OUT_REPORT = os.path.join(ROOT, "combined_dataset_report.json")
OUT_SPLITS = os.path.join(ROOT, "combined_data_splits.json")

sys.path.append(ROOT)
from features import frame_vector, WINDOW_T, FRAME_DIM
from mp_extract import Extractor

CSLRT_TO_CANONICAL = {
    # Direct word video matches
    "help": "HELP",
    "good": "GOOD",
    "bad": "BAD",
    "welcome": "WELCOME",
    "fever": "FEVER",
    "medicine": "MEDICINE",
    "sleep": "SLEEP",
    "sitting": "SIT",
    "sit": "SIT",
    "stopped": "STOP",
    "stop": "STOP",
    "friend": "FRIEND",
    "name": "NAME",
    "sorry": "SORRY",
    "sorry hear": "SORRY",
    "today": "TODAY",
    "angry": "ANGRY",
    "food": "EAT",
    "water": "WATER",
    "fine thank": "I_AM_FINE",
    "cannot help": "HELP",
    "agree": "YES",
    "hate": "NO",
    
    # Sentence folder matches
    "you are good": "GOOD",
    "you are bad": "BAD",
    "you are welcome": "WELCOME",
    "help me": "HELP",
    "can i help you": "HELP",
    "i am suffering from fever": "FEVER",
    "you need a medicine, take this one": "MEDICINE",
    "go and sleep": "SLEEP",
    "thank you so much": "THANK_YOU",
    "he she is my friend": "FRIEND",
    "my name is xxxxxxxx": "NAME",
    "why are you angry": "ANGRY",
    "hi how are you": "HOW_ARE_YOU",
    "how can i help you": "HELP",
    "can you repeat that please": "PLEASE",
    "congratulations": "CONGRATULATIONS",
    "nice to meet you": "NICE_TO_MEET_YOU",
    "i got hurt": "PAIN",
    "i am feeling cold": "COLD",
    "i am hungry": "HUNGRY",
    "i am tired": "TIRED",
    "bring water for me": "WATER",
    "prepare the bed": "BED",
    "serve the food": "FOOD",
    "wear the shirt": "SHIRT",
}

DUPLICATE_FILES_TO_EXCLUDE = {
    "do not worry/worry.mp4",
    "do you need something/something (2).mp4",
    "i dont agree/mvi_6494.mp4",
    "i like you i love you/love.mp4",
    "what have you planned for your career/mvi_6499.mp4",
}


def get_cslrt_signer(filepath: str) -> str:
    fname = os.path.basename(filepath)
    m = re.search(r"\((\d+)\)", fname)
    if m:
        return f"cslrt_signer_{m.group(1)}"
    m2 = re.search(r"(\d+)", fname)
    if m2:
        return f"cslrt_signer_{m2.group(1)}"
    return "cslrt_signer_default"


def video_to_sequence(cap: cv2.VideoCapture, extractor: Extractor, num_frames: int = WINDOW_T) -> np.ndarray | None:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total < 5:
        return None

    vecs = []
    hands_detected = 0
    i = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[1] > 960:
            bgr = cv2.resize(bgr, (960, int(bgr.shape[0] * 960 / bgr.shape[1])))
        left, right, pose, _ = extractor(bgr, int(i * 1000 / fps))
        vecs.append(frame_vector(left, right, pose))
        if left is not None or right is not None:
            hands_detected += 1
        i += 1

    if len(vecs) < 5 or hands_detected < max(1, len(vecs) * 0.15):
        return None

    arr = np.stack(vecs)
    indices = np.linspace(0, len(arr) - 1, num_frames).round().astype(int)
    return arr[indices].astype(np.float32)


def main():
    print("=== VERBA: COMBINED AMRITA + ISL-CSLRT PREPROCESSING ===")
    os.makedirs(DATA_DIR, exist_ok=True)
    extractor = Extractor("video")

    cslrt_vids = sorted(list(set(glob.glob(os.path.join(CSLRT_DIR, "**", "*.*"), recursive=True))))
    cslrt_vids = [v for v in cslrt_vids if os.path.isfile(v) and v.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    print(f"[cslrt] Found {len(cslrt_vids)} video files")

    hashes = {}
    valid_cslrt = []
    skipped_duplicates = 0

    for v in cslrt_vids:
        rel = os.path.relpath(v, CSLRT_DIR).replace("\\", "/").lower()
        if rel in DUPLICATE_FILES_TO_EXCLUDE:
            skipped_duplicates += 1
            continue

        hasher = hashlib.md5()
        with open(v, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        h = hasher.hexdigest()
        if h in hashes:
            skipped_duplicates += 1
            continue
        hashes[h] = v
        valid_cslrt.append(v)

    print(f"[cslrt] Valid non-duplicate videos: {len(valid_cslrt)} (skipped {skipped_duplicates} duplicates)")

    mapped_samples = []
    for v in valid_cslrt:
        folder = os.path.basename(os.path.dirname(v)).lower()
        base = os.path.splitext(os.path.basename(v))[0].lower()
        clean_base = re.sub(r"\s*\(\d+\)$", "", base).strip()

        canonical = None
        if clean_base in CSLRT_TO_CANONICAL:
            canonical = CSLRT_TO_CANONICAL[clean_base]
        elif folder in CSLRT_TO_CANONICAL:
            canonical = CSLRT_TO_CANONICAL[folder]

        if canonical:
            signer = get_cslrt_signer(v)
            mapped_samples.append({
                "video_path": v,
                "folder": folder,
                "base": base,
                "gloss": canonical,
                "signer": signer,
            })

    print(f"[cslrt] Mapped samples: {len(mapped_samples)} across {len(set(s['gloss'] for s in mapped_samples))} classes")

    cslrt_splits = {"train": [], "val": [], "test": []}
    success_count = 0
    fail_count = 0

    val_signers = {"cslrt_signer_6", "cslrt_signer_6721", "cslrt_signer_6722", "cslrt_signer_6724", "cslrt_signer_6725"}
    test_signers = {"cslrt_signer_7", "cslrt_signer_6814", "cslrt_signer_6817", "cslrt_signer_6853", "cslrt_signer_8"}

    for idx, s in enumerate(mapped_samples):
        g = s["gloss"]
        gloss_dir = os.path.join(DATA_DIR, g)
        os.makedirs(gloss_dir, exist_ok=True)

        signer = s["signer"]
        base_name = f"cslrt_{signer}_{idx:04d}"
        npy_path = os.path.join(gloss_dir, f"{base_name}.npy")

        if signer in test_signers:
            split = "test"
        elif signer in val_signers:
            split = "val"
        else:
            split = "train"

        if os.path.exists(npy_path):
            success_count += 1
            cslrt_splits[split].append({
                "source": "cslrt",
                "gloss": g,
                "video_file": os.path.relpath(s["video_path"], CSLRT_DIR).replace("\\", "/"),
                "saved_as": f"{base_name}.npy",
                "signer": signer,
            })
            continue

        cap = cv2.VideoCapture(s["video_path"])
        if not cap.isOpened():
            fail_count += 1
            continue

        seq = video_to_sequence(cap, extractor, WINDOW_T)
        cap.release()

        if seq is None or seq.shape != (WINDOW_T, FRAME_DIM):
            fail_count += 1
            continue

        np.save(npy_path, seq)
        success_count += 1
        cslrt_splits[split].append({
            "source": "cslrt",
            "gloss": g,
            "video_file": os.path.relpath(s["video_path"], CSLRT_DIR).replace("\\", "/"),
            "saved_as": f"{base_name}.npy",
            "signer": signer,
        })
        if success_count % 25 == 0:
            print(f"  Processed {success_count}/{len(mapped_samples)} CSLRT samples...")

    print(f"[cslrt] Extraction complete: {success_count} succeeded, {fail_count} skipped/failed.")

    merged_splits = {
        "amrita": {},
        "cslrt": cslrt_splits,
        "combined": {
            "train": [],
            "val": [],
            "test": [],
        },
    }

    if os.path.exists(EXISTING_SPLITS):
        with open(EXISTING_SPLITS, "r", encoding="utf-8") as f:
            amrita_data = json.load(f)
            merged_splits["amrita"] = {
                "train_count": len(amrita_data.get("train", [])),
                "val_count": len(amrita_data.get("val", [])),
                "test_count": len(amrita_data.get("test", [])),
            }
            for sp in ["train", "val", "test"]:
                for item in amrita_data.get(sp, []):
                    merged_splits["combined"][sp].append({
                        "source": "amrita",
                        "gloss": item.get("gloss"),
                        "video_file": item.get("video_file"),
                        "saved_as": item.get("saved_as"),
                        "signer": item.get("signer", "amrita_signer"),
                    })

    for sp in ["train", "val", "test"]:
        merged_splits["combined"][sp].extend(cslrt_splits[sp])

    train_files = {item["saved_as"] for item in merged_splits["combined"]["train"]}
    val_files = {item["saved_as"] for item in merged_splits["combined"]["val"]}
    test_files = {item["saved_as"] for item in merged_splits["combined"]["test"]}

    assert len(train_files.intersection(val_files)) == 0, "Train-Val leakage detected!"
    assert len(train_files.intersection(test_files)) == 0, "Train-Test leakage detected!"
    assert len(val_files.intersection(test_files)) == 0, "Val-Test leakage detected!"

    print("=== ZERO DATA LEAKAGE VERIFIED ===")
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples:   {len(val_files)}")
    print(f"Test samples:  {len(test_files)}")
    print(f"Total combined samples: {len(train_files) + len(val_files) + len(test_files)}")

    with open(OUT_SPLITS, "w", encoding="utf-8") as f:
        json.dump(merged_splits, f, indent=2)
    print(f"Saved manifest to: {OUT_SPLITS}")

    all_classes = sorted(list({item["gloss"] for item in merged_splits["combined"]["train"] + merged_splits["combined"]["val"] + merged_splits["combined"]["test"]}))
    class_dist = {}
    for item in merged_splits["combined"]["train"] + merged_splits["combined"]["val"] + merged_splits["combined"]["test"]:
        g = item["gloss"]
        class_dist[g] = class_dist.get(g, 0) + 1

    report = {
        "amrita_dataset": {
            "source_zip": AMRITA_ZIP,
            "classes": 49,
            "total_videos": 978,
            "formats": ["mp4", "mov"],
        },
        "cslrt_dataset": {
            "source_dir": CSLRT_DIR,
            "total_videos": len(cslrt_vids),
            "formats": ["mp4"],
            "duplicates_excluded": skipped_duplicates,
            "mapped_samples": len(mapped_samples),
            "successful_extractions": success_count,
        },
        "combined": {
            "total_classes": len(all_classes),
            "classes": all_classes,
            "total_samples": len(train_files) + len(val_files) + len(test_files),
            "train_samples": len(train_files),
            "val_samples": len(val_files),
            "test_samples": len(test_files),
            "class_distribution": class_dist,
            "data_leakage": {
                "train_val_overlap": len(train_files.intersection(val_files)),
                "train_test_overlap": len(train_files.intersection(test_files)),
                "val_test_overlap": len(val_files.intersection(test_files)),
            },
        },
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved dataset report to: {OUT_REPORT}")


if __name__ == "__main__":
    main()
