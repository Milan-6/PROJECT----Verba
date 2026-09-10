"""
Programmatic inspection of the Amrita SLR ISL dataset inside the repository.
Analyzes classes, video formats, corrupt/unreadable samples, durations, frame counts, and resolutions.
Generates ml_pipeline/dataset_report.json.
"""
from __future__ import annotations
import collections
import json
import os
import sys
import tempfile
import zipfile
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.abspath(os.path.join(ROOT, "..", "dataset", "Amrita SLR- Real Time ISL Dataset", "Amrita SLR Dataset.zip"))
REPORT_PATH = os.path.join(ROOT, "dataset_report.json")


def inspect_dataset(zip_path: str = ZIP_PATH, report_path: str = REPORT_PATH):
    if not os.path.exists(zip_path):
        sys.exit(f"Error: Amrita dataset zip not found at {zip_path}")

    print(f"Inspecting Amrita dataset at:\n  {zip_path}\n")
    z = zipfile.ZipFile(zip_path)
    file_list = [f for f in z.namelist() if not f.endswith("/")]

    classes_dict = collections.defaultdict(list)
    ext_counter = collections.Counter()
    corrupt_videos = []

    # Map files to classes
    for item in file_list:
        parts = item.split("/")
        if len(parts) >= 3:
            cls_name = parts[1]
            video_name = parts[2]
            classes_dict[cls_name].append(item)
            ext = video_name.split(".")[-1].lower() if "." in video_name else "none"
            ext_counter[ext] += 1

    print(f"Total entries in archive: {len(file_list)}")
    print(f"Detected classes: {len(classes_dict)}")
    print(f"Video extensions: {dict(ext_counter)}")

    # Sample and verify video readability, resolution, fps, duration
    print("\nVerifying video readability and sampling properties across all videos...")
    resolutions = collections.Counter()
    fps_list = []
    durations = []
    frame_counts = []

    with tempfile.TemporaryDirectory() as td:
        for cls_name, items in sorted(classes_dict.items()):
            for item in items:
                try:
                    extracted_path = z.extract(item, td)
                    cap = cv2.VideoCapture(extracted_path)
                    if not cap.isOpened():
                        corrupt_videos.append(item)
                        cap.release()
                        try: os.remove(extracted_path)
                        except Exception: pass
                        continue
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0.0

                    if total_frames < 3 or w == 0 or h == 0:
                        corrupt_videos.append(item)
                    else:
                        resolutions[f"{w}x{h}"] += 1
                        fps_list.append(round(fps, 1))
                        durations.append(round(duration, 2))
                        frame_counts.append(total_frames)

                    cap.release()
                    try: os.remove(extracted_path)
                    except Exception: pass
                except Exception as e:
                    corrupt_videos.append(f"{item} (error: {e})")

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
    avg_dur = sum(durations) / len(durations) if durations else 0.0
    avg_frames = sum(frame_counts) / len(frame_counts) if frame_counts else 0.0

    report = {
        "dataset_path": zip_path,
        "dataset_size_bytes": os.path.getsize(zip_path),
        "num_classes": len(classes_dict),
        "num_videos": sum(len(v) for v in classes_dict.values()),
        "extensions": dict(ext_counter),
        "corrupt_videos": corrupt_videos,
        "empty_classes": [c for c, v in classes_dict.items() if len(v) == 0],
        "resolutions": dict(resolutions),
        "average_fps": round(avg_fps, 2),
        "average_duration_seconds": round(avg_dur, 2),
        "average_frame_count": round(avg_frames, 1),
        "min_duration_seconds": min(durations) if durations else 0.0,
        "max_duration_seconds": max(durations) if durations else 0.0,
        "classes": {c: len(v) for c, v in sorted(classes_dict.items())}
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {report_path}")
    print(f"Classes: {report['num_classes']}")
    print(f"Total Videos: {report['num_videos']}")
    print(f"Corrupt Videos: {len(corrupt_videos)}")
    print(f"Resolutions: {report['resolutions']}")
    print(f"Avg Duration: {report['average_duration_seconds']}s (range: {report['min_duration_seconds']}s - {report['max_duration_seconds']}s)")
    print(f"Avg FPS: {report['average_fps']}")
    return report


if __name__ == "__main__":
    inspect_dataset()
