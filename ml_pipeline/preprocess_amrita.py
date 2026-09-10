"""
Preprocesses the Amrita SLR ISL dataset into training-ready landmark sequences.
- Extracts videos from Amrita SLR Dataset.zip without modifying original data.
- Video-level splitting: Split is strictly by video file (train/val/test).
- Verifies zero data leakage (train ∩ val = 0, train ∩ test = 0, val ∩ test = 0).
- Normalizes landmarks using body scale and limb vectors (30 frames, 270 features).
- Caches preprocessed sequences in ml_pipeline/data/<GLOSS>/amrita_<split>_<id>.npy.
- Saves splitting manifest in ml_pipeline/data_splits.json.
"""
from __future__ import annotations
import collections
import json
import os
import random
import shutil
import tempfile
import zipfile
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.abspath(os.path.join(ROOT, "..", "dataset", "Amrita SLR- Real Time ISL Dataset", "Amrita SLR Dataset.zip"))
DATA_DIR = os.path.join(ROOT, "data")
SPLITS_PATH = os.path.join(ROOT, "data_splits.json")

# Import feature pipeline from features.py and mp_extract.py
from features import frame_vector, WINDOW_T, FRAME_DIM
from mp_extract import Extractor

# Mapping from Amrita class folder to normalized uppercase gloss
AMRITA_GLOSS_MAP = {
    "9.hello": "HELLO",
    "24.help": "HELP",
    "7.goodmorning": "GOOD_MORNING",
    "38.thankyou": "THANK_YOU",
    "32.sorry": "SORRY",
    "31.sit": "SIT",
    "26.NAME": "NAME",
    "46.write": "WRITE",
    "16.eat": "EAT",
    "18.drinking": "DRINK",
    "30.sleep": "SLEEP",
    "33.stand": "STAND",
    "34.stop": "STOP",
    "39.today": "TODAY",
    "43.yesterday": "YESTERDAY",
    "41.welcome": "WELCOME",
    "10.youareperfect": "YOU_ARE_PERFECT",
    "17.friend": "FRIEND",
    "21.good": "GOOD",
    "14.bad": "BAD",
    "42.work": "WORK",
    "15.boy": "BOY",
    "19.girl": "GIRL",
    "22.father": "FATHER",
    "25.mother": "MOTHER",
    "20.brother": "BROTHER",
}


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

    if len(vecs) < 5 or hands_detected < len(vecs) * 0.2:
        return None

    arr = np.stack(vecs)
    indices = np.linspace(0, len(arr) - 1, num_frames).round().astype(int)
    return arr[indices].astype(np.float32)


def preprocess_amrita(
    zip_path: str = ZIP_PATH,
    classes_to_process: dict[str, str] = AMRITA_GLOSS_MAP,
    samples_per_class: int = 15,
    seed: int = 42,
):
    print("=== AMRITA SLR PREPROCESSING PIPELINE ===")
    print(f"Reading from: {zip_path}")
    random.seed(seed)
    np.random.seed(seed)

    z = zipfile.ZipFile(zip_path)
    all_files = z.namelist()

    extractor = Extractor("video")
    splits = {
        "seed": seed,
        "train": [],
        "val": [],
        "test": [],
        "leakage_check": {},
    }

    total_processed = 0
    total_skipped = 0

    with tempfile.TemporaryDirectory() as td:
        for amrita_cls, gloss in sorted(classes_to_process.items()):
            # Find all videos for this class in the zip
            vids = sorted([f for f in all_files if amrita_cls in f and f.lower().endswith((".mp4", ".mov"))])
            if not vids:
                continue

            # Video-level split: Shuffle video filenames
            vids_shuffled = list(vids)
            random.shuffle(vids_shuffled)
            selected_vids = vids_shuffled[:samples_per_class]

            n = len(selected_vids)
            n_train = max(1, int(n * 0.70))
            n_val = max(1, int(n * 0.15))
            train_files = selected_vids[:n_train]
            val_files = selected_vids[n_train : n_train + n_val]
            test_files = selected_vids[n_train + n_val :]

            # Data leakage check at the video level
            tr_set, val_set, te_set = set(train_files), set(val_files), set(test_files)
            assert len(tr_set & val_set) == 0, f"Data leakage between train and val in {gloss}!"
            assert len(tr_set & te_set) == 0, f"Data leakage between train and test in {gloss}!"
            assert len(val_set & te_set) == 0, f"Data leakage between val and test in {gloss}!"

            out_dir = os.path.join(DATA_DIR, gloss)
            os.makedirs(out_dir, exist_ok=True)

            kept_class = 0
            for split_name, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
                for idx, vfile in enumerate(flist):
                    try:
                        extracted_video = z.extract(vfile, td)
                        cap = cv2.VideoCapture(extracted_video)
                        seq = video_to_sequence(cap, extractor, WINDOW_T)
                        cap.release()
                        os.remove(extracted_video)

                        if seq is not None and seq.shape == (WINDOW_T, FRAME_DIM):
                            save_name = f"amrita_{split_name}_{idx}.npy"
                            save_path = os.path.join(out_dir, save_name)
                            np.save(save_path, seq)
                            splits[split_name].append({
                                "video_file": vfile,
                                "gloss": gloss,
                                "saved_as": save_name,
                            })
                            kept_class += 1
                            total_processed += 1
                        else:
                            total_skipped += 1
                    except Exception as e:
                        total_skipped += 1
                        print(f"  [skip] Error processing {vfile}: {e}")

            print(f"  [OK] {gloss:<16} ({amrita_cls}): {kept_class}/{len(selected_vids)} sequences extracted")

    extractor.close()

    # Verify overall leakage
    tr_all = {item["video_file"] for item in splits["train"]}
    val_all = {item["video_file"] for item in splits["val"]}
    test_all = {item["video_file"] for item in splits["test"]}

    leakage_tr_val = len(tr_all & val_all)
    leakage_tr_test = len(tr_all & test_all)
    leakage_val_test = len(val_all & test_all)

    splits["leakage_check"] = {
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "overlap_train_val": leakage_tr_val,
        "overlap_train_test": leakage_tr_test,
        "overlap_val_test": leakage_val_test,
        "data_leakage_detected": (leakage_tr_val + leakage_tr_test + leakage_val_test) > 0,
    }

    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"Total processed sequences: {total_processed} (skipped/untrackable: {total_skipped})")
    print(f"Train samples: {len(splits['train'])}")
    print(f"Validation samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    print(f"Leakage Check: train intersect val = {leakage_tr_val}, train intersect test = {leakage_tr_test}, val intersect test = {leakage_val_test}")
    print(f"Saved manifest to: {SPLITS_PATH}")
    return splits


if __name__ == "__main__":
    preprocess_amrita()
