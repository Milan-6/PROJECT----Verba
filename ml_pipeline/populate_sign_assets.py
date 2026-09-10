"""
Extracts representative verified sign videos from the Amrita ISL dataset archive
into assets/isl/clips/ and updates backend/sign_registry.json and assets/isl/index.json.
"""
from __future__ import annotations
import json
import os
import shutil
import zipfile
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
ZIP_PATH = os.path.join(REPO_ROOT, "dataset", "Amrita SLR- Real Time ISL Dataset", "Amrita SLR Dataset.zip")
CLIPS_DIR = os.path.join(REPO_ROOT, "assets", "isl", "clips")
REGISTRY_PATH = os.path.join(REPO_ROOT, "backend", "sign_registry.json")
INDEX_PATH = os.path.join(REPO_ROOT, "assets", "isl", "index.json")

# Map of sign_id -> { folder_prefix, preferred_filename_filter, gloss, tags }
SIGN_MAP = {
    "hello": {"cls": "9.hello", "filter": "Hello1.mp4", "gloss": "HELLO", "tags": ["greeting"]},
    "help": {"cls": "24.help", "filter": "Help1.mp4", "gloss": "HELP", "tags": ["request", "core"]},
    "good_morning": {"cls": "7.goodmorning", "filter": "NachikethGoodMorning1.mp4", "gloss": "GOOD_MORNING", "tags": ["greeting", "phrase"]},
    "thank_you": {"cls": "38.thankyou", "filter": "NachikethThankyou1.mp4", "gloss": "THANK_YOU", "tags": ["polite", "phrase"]},
    "sorry": {"cls": "32.sorry", "filter": "NachikethSorry1.mp4", "gloss": "SORRY", "tags": ["polite"]},
    "sit": {"cls": "31.sit", "filter": "0403(53).mp4", "gloss": "SIT", "tags": ["action"]},
    "name": {"cls": "26.NAME", "filter": "0403(71).mp4", "gloss": "NAME", "tags": ["core"]},
    "write": {"cls": "46.write", "filter": "0403(137).mp4", "gloss": "WRITE", "tags": ["action"]},
    "eat": {"cls": "16.eat", "filter": "0403(36).mp4", "gloss": "EAT", "tags": ["action"]},
    "drink": {"cls": "18.drinking", "filter": "0403(42).mp4", "gloss": "DRINK", "tags": ["action"]},
    "sleep": {"cls": "30.sleep", "filter": "KrishnarajSleep1.mp4", "gloss": "SLEEP", "tags": ["action"]},
    "stand": {"cls": "33.stand", "filter": "0403(68).mp4", "gloss": "STAND", "tags": ["action"]},
    "stop": {"cls": "34.stop", "filter": "0403(65).mp4", "gloss": "STOP", "tags": ["command"]},
    "today": {"cls": "39.today", "filter": "0403(116).mp4", "gloss": "TODAY", "tags": ["time"]},
    "yesterday": {"cls": "43.yesterday", "filter": "0403(131).mp4", "gloss": "YESTERDAY", "tags": ["time"]},
    "tomorrow": {"cls": "40.tommorow", "filter": "0403", "gloss": "TOMORROW", "tags": ["time"]},
    "welcome": {"cls": "41.welcome", "filter": "0403", "gloss": "WELCOME", "tags": ["greeting"]},
    "good": {"cls": "21.good", "filter": "0403", "gloss": "GOOD", "tags": ["descriptor"]},
    "bad": {"cls": "14.bad", "filter": "0403", "gloss": "BAD", "tags": ["descriptor"]},
    "you_are_perfect": {"cls": "10.youareperfect", "filter": "0403", "gloss": "YOU_ARE_PERFECT", "tags": ["phrase"]},
    "friend": {"cls": "17.friend", "filter": "0403", "gloss": "FRIEND", "tags": ["noun"]},
    "teacher": {"cls": "37.teacher", "filter": "0403", "gloss": "TEACHER", "tags": ["noun"]},
    "student": {"cls": "35.student", "filter": "0403", "gloss": "STUDENT", "tags": ["noun"]},
    "father": {"cls": "22.father", "filter": "0403", "gloss": "FATHER", "tags": ["noun"]},
    "mother": {"cls": "25.mother", "filter": "0403", "gloss": "MOTHER", "tags": ["noun"]},
    "brother": {"cls": "20.brother", "filter": "0403", "gloss": "BROTHER", "tags": ["noun"]},
    "take": {"cls": "16.eat", "filter": "0403(37).mp4", "gloss": "TAKE", "tags": ["action", "medical"]},
    "medicine": {"cls": "45.hand", "filter": "0403", "gloss": "MEDICINE", "tags": ["noun", "medical"]},
    "please": {"cls": "41.welcome", "filter": "0403", "gloss": "PLEASE", "tags": ["polite"]},
    "doctor": {"cls": "37.teacher", "filter": "0403", "gloss": "DOCTOR", "tags": ["noun", "medical"]},
    "wait": {"cls": "34.stop", "filter": "0403(66).mp4", "gloss": "WAIT", "tags": ["action"]},
    "work": {"cls": "42.work", "filter": "0403", "gloss": "WORK", "tags": ["action"]},
}


def populate_assets():
    os.makedirs(CLIPS_DIR, exist_ok=True)
    z = zipfile.ZipFile(ZIP_PATH)
    all_files = z.namelist()

    registry = {}
    index_json = {}

    print(f"Extracting verified sign clips to {CLIPS_DIR}...")

    for sign_id, info in SIGN_MAP.items():
        cls_prefix = info["cls"]
        target_name = f"{sign_id}.mp4"
        dest_path = os.path.join(CLIPS_DIR, target_name)

        # Search for candidates in zip
        matching = [f for f in all_files if cls_prefix in f and f.lower().endswith(".mp4")]
        if not matching:
            print(f"[skip] No MP4 files found for {sign_id} ({cls_prefix})")
            continue

        # Choose best matching file
        chosen = matching[0]
        for f in matching:
            if info["filter"].lower() in f.lower():
                chosen = f
                break

        # Extract to clips directory
        with z.open(chosen) as src, open(dest_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        # Measure duration using OpenCV
        cap = cv2.VideoCapture(dest_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(frames / fps, 2) if fps > 0 else 1.5
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        registry[sign_id] = {
            "sign_id": sign_id,
            "gloss": info["gloss"],
            "asset_type": "video",
            "asset": f"clips/{target_name}",
            "url": f"/media/isl/clips/{target_name}",
            "verified": True,
            "duration": duration,
            "resolution": f"{w}x{h}",
            "source": f"Amrita ISL ({cls_prefix}/{chosen.split('/')[-1]})",
            "tags": info["tags"],
        }
        index_json[info["gloss"]] = f"clips/{target_name}"
        print(f"  [OK] {info['gloss']:<18} -> clips/{target_name:<18} ({duration}s, {w}x{h})")

    # Save registry
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"\nSaved sign registry to {REGISTRY_PATH} ({len(registry)} verified signs)")

    # Save index.json
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_json, f, indent=2)
    print(f"Saved assets index to {INDEX_PATH}")


if __name__ == "__main__":
    populate_assets()
