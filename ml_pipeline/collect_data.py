"""
Record training sequences from the webcam. Saves ONLY landmark vectors, never video.

  python collect_data.py --gloss PAIN --signer bhargav --samples 35
  python collect_data.py --gloss NONE --signer bhargav --samples 60      # idle data
  python collect_data.py --scenario bank --signer priya --samples 30     # loops over every gloss

Output: data/<GLOSS>/<signer>_<n>.npy   shape (30, 270)

Keys during capture:  SPACE = start next sample   a = toggle auto-repeat   q = quit   s = skip gloss

  --auto            record sample after sample without pressing SPACE each time (fastest way
                    to collect 30 samples: just repeat the sign, returning to rest between each)
  --gap 1.5         seconds of "GET READY" between samples in auto mode
Tips: two distances (1 m / 2 m), two lighting setups, both hands visible,
      start and END each sample in the rest position (hands down).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import cv2
from features import frame_vector, WINDOW_T, FRAME_DIM
from mp_extract import Extractor, draw

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
VOCAB = os.path.join(ROOT, "..", "backend", "vocab")


def load_glosses(scenario: str) -> list[str]:
    with open(os.path.join(VOCAB, f"{scenario}.json")) as f:
        v = json.load(f)
    return v["core"] + v["glosses"]


def next_index(gloss: str, signer: str) -> int:
    d = os.path.join(DATA, gloss)
    os.makedirs(d, exist_ok=True)
    return len([f for f in os.listdir(d) if f.startswith(signer + "_")])


def record_gloss(cap, ex, gloss: str, signer: str, samples: int, frames: int,
                 camera_mirror=True, auto: bool = False, gap: float = 1.5) -> bool:
    n0 = next_index(gloss, signer)
    n = 0
    state = "wait"       # wait -> countdown -> capture
    buf, t_state = [], time.time()
    while n < samples:
        ok, bgr = cap.read()
        if not ok:
            print("camera read failed"); return False
        # IMPORTANT: extract from the RAW frame (the browser does the same); mirror only for display
        left, right, pose, _ = ex(bgr)
        if camera_mirror:
            bgr = cv2.flip(bgr, 1)
        draw(bgr, left, right, pose, mirror=camera_mirror)
        both = left is not None and right is not None
        q = "both hands" if both else ("one hand" if (left is not None or right is not None) else "NO HANDS")
        col = (0, 255, 120) if both else ((0, 200, 255) if q == "one hand" else (0, 0, 255))
        cv2.putText(bgr, f"{gloss}  sample {n0 + n + 1}/{n0 + samples}  signer={signer}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(bgr, f"tracking: {q}   pose: {'ok' if pose is not None else 'NONE'}"
                        + ("   AUTO" if auto else ""), (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        if state == "wait":
            msg = ("SPACE = record   a = auto-repeat   s = skip gloss   q = quit" if gloss != "NONE"
                   else "NONE mode: SPACE starts continuous idle recording")
            cv2.putText(bgr, msg, (12, bgr.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        elif state == "countdown":
            left_s = gap - (time.time() - t_state)
            cv2.putText(bgr, f"GET READY {max(0, left_s):.1f}", (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
            if left_s <= 0:
                state, buf = "capture", []
        elif state == "capture":
            buf.append(frame_vector(left, right, pose))
            cv2.putText(bgr, f"RECORDING {len(buf)}/{frames}", (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if len(buf) >= frames:
                arr = np.stack(buf).astype(np.float32)
                assert arr.shape == (frames, FRAME_DIM)
                np.save(os.path.join(DATA, gloss, f"{signer}_{n0 + n}.npy"), arr)
                n += 1
                if gloss == "NONE":
                    state, buf = "capture", []          # continuous idle chunks
                elif auto:
                    state, t_state = "countdown", time.time()   # keep going: sign, rest, sign, rest
                else:
                    state, t_state = "wait", time.time()

        cv2.imshow("SignBridge collect", bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            return False
        if k == ord('s'):
            return True
        if k == ord('a') and gloss != "NONE":
            auto = not auto
            if auto and state == "wait":
                state, t_state = "countdown", time.time()
        if k == ord(' ') and state == "wait":
            state, t_state = "countdown", time.time()
    print(f"[ok] {gloss}: saved {n} samples")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gloss")
    ap.add_argument("--scenario", help="loop over every gloss in backend/vocab/<scenario>.json")
    ap.add_argument("--signer", required=True)
    ap.add_argument("--samples", type=int, default=35)
    ap.add_argument("--frames", type=int, default=WINDOW_T)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--auto", action="store_true", help="record back-to-back without pressing SPACE")
    ap.add_argument("--gap", type=float, default=1.5, help="seconds between samples in auto mode")
    args = ap.parse_args()
    glosses = [args.gloss] if args.gloss else load_glosses(args.scenario)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        raise SystemExit("cannot open camera; try --camera 1")
    ex = Extractor("video")
    try:
        for g in glosses:
            if not record_gloss(cap, ex, g, args.signer, args.samples, args.frames,
                                auto=args.auto, gap=args.gap):
                break
    finally:
        ex.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
