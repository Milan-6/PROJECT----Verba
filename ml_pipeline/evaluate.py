"""
Evaluate the exported model the way the demo will use it.

  python evaluate.py --holdout priya

1) confusion matrix on the held-out signer (saved to backend/models/confusion.png)
2) IDLE FALSE-FIRE TEST: stream every NONE sequence through the real commit logic
   and count candidates fired. Target = 0. This is the number that decides whether
   the app spits out random words while someone scratches their nose on stage.
3) live-style accuracy: stream each held-out sequence (padded with rest) through the
   recognizer and check the first candidate fired equals the label.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "backend"))
from inference import SignRecognizer, STRIDE  # noqa: E402
from features import FRAME_DIM  # noqa: E402

DATA = os.path.join(ROOT, "data")


REST = None   # real NONE frames used to pad sequences (set in main)


def stream(rec: SignRecognizer, seq: np.ndarray):
    rec.reset()
    rest = REST if REST is not None else np.zeros((10, FRAME_DIM), np.float32)
    fired = []
    t = 0.0
    for v in np.concatenate([rest, seq, rest]):
        t += 1 / 30
        r = rec.push(v, now=t)
        if r and r[0] == "candidate":
            fired.append(r[1])
    return fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--models", default=None)
    args = ap.parse_args()
    rec = SignRecognizer(args.models) if args.models else SignRecognizer()
    labels = rec.labels
    L = {g: i for i, g in enumerate(labels)}

    # ---- idle false-fire test --------------------------------------------
    none_files = glob.glob(os.path.join(args.data, "NONE", "*.npy"))
    global REST
    if none_files:
        REST = np.load(none_files[0])[:15]
    fires = 0
    for f in none_files:
        fires += len(stream(rec, np.load(f)))
    print(f"[idle test] {len(none_files)} NONE sequences -> {fires} false candidates  "
          f"({'PASS' if fires == 0 else 'FAIL: raise THRESH/CONSEC or record more NONE'})")

    # ---- live-style accuracy + confusion on held-out signer --------------
    cm = np.zeros((len(labels), len(labels) + 1), int)   # extra col = nothing fired
    n = hit = 0
    for g in labels:
        if g == "NONE":
            continue
        for f in glob.glob(os.path.join(args.data, g, f"{args.holdout}_*.npy")):
            fired = stream(rec, np.load(f))
            n += 1
            if fired:
                cm[L[g], L[fired[0]]] += 1; hit += int(fired[0] == g)
            else:
                cm[L[g], -1] += 1
    if n == 0:
        sys.exit(f"no sequences for signer {args.holdout}")
    print(f"[live-style] first candidate == label: {hit}/{n} = {hit / n:.3f}   "
          f"(nothing fired: {cm[:, -1].sum()})")
    worst = sorted(((cm[i, i] / max(cm[i].sum(), 1), g) for i, g in enumerate(labels) if g != "NONE"))[:6]
    print("weakest:", ", ".join(f"{g} {a:.2f}" for a, g in worst))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels) + 1)); ax.set_xticklabels(labels + ["(none)"], rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("first candidate fired"); ax.set_ylabel("true sign")
        out = os.path.join(ROOT, "..", "backend", "models", "confusion.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); print("saved", out)
    except Exception as e:  # matplotlib optional
        print("(no confusion plot:", e, ")")


if __name__ == "__main__":
    main()
