"""
Convert sign videos (e.g. the IIT-Madras INCLUDE dataset, which is the public release of the
"General Dataset" in the thesis, or your own phone recordings) into training sequences.

Expected layout:   <root>/<GLOSS or word>/<any>.mp4      (one folder per sign)
Output:            data/<GLOSS>/<signer>_<n>.npy   (30, 270)

  python preprocess_video.py --root ~/INCLUDE --map include_map.json --signer include
  python preprocess_video.py --root ./my_clips --signer bhargav_phone

--map is an optional JSON {"Balance": "BALANCE", "Thank You": "THANK_YOU", ...} that renames
dataset folders to your vocabulary glosses; folders not in the map are skipped.
Videos of 1-4 s are resampled to 30 evenly spaced frames (thesis: 25-60 fps, 60-90 frames).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import cv2
from features import frame_vector, RECORD_T, FRAME_DIM
from mp_extract import Extractor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


def video_to_seq(path: str, frames: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total < 5:
        cap.release(); return None
    ex = Extractor("video")
    vecs, got = [], 0
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[1] > 960:
            bgr = cv2.resize(bgr, (960, int(bgr.shape[0] * 960 / bgr.shape[1])))
        l, r, p, _ = ex(bgr, int(i * 1000 / fps))
        vecs.append(frame_vector(l, r, p))
        got += int(l is not None or r is not None)
        i += 1
    ex.close(); cap.release()
    if len(vecs) < 5 or got < len(vecs) * 0.3:
        return None                                   # hands never tracked -> useless clip
    arr = np.stack(vecs)
    idx = np.linspace(0, len(arr) - 1, frames).round().astype(int)
    return arr[idx].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--signer", required=True)
    ap.add_argument("--map")
    ap.add_argument("--frames", type=int, default=RECORD_T)
    args = ap.parse_args()
    name_map = json.load(open(args.map)) if args.map else None
    folders = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    for folder in folders:
        gloss = (name_map or {}).get(folder, None if name_map else folder.upper().replace(" ", "_"))
        if gloss is None:
            continue
        vids = sorted(glob.glob(os.path.join(args.root, folder, "*.*")))
        out_dir = os.path.join(DATA, gloss); os.makedirs(out_dir, exist_ok=True)
        n = len([f for f in os.listdir(out_dir) if f.startswith(args.signer + "_")])
        kept = 0
        for v in vids:
            seq = video_to_seq(v, args.frames)
            if seq is None:
                continue
            np.save(os.path.join(out_dir, f"{args.signer}_{n + kept}.npy"), seq); kept += 1
        print(f"{folder:>20} -> {gloss:<16} {kept}/{len(vids)} clips")


if __name__ == "__main__":
    main()
