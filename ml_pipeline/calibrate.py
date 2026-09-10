"""
Find the confidence threshold that actually suits YOUR model, instead of guessing one.

  python calibrate.py --holdout bhargav

Why this exists: the right threshold depends on how many classes you trained. With 7 classes
a softmax of 0.75 is common; with 12+ the probability mass spreads thinner and the same 0.75
means most signs never fire at all — they go silent. That is the "often nothing fires" symptom.

The sweep runs your real recordings through the real live commit logic for every
(threshold, consecutive-windows) pair, and picks the setting with the best live accuracy
that STILL fires zero candidates on your idle NONE recordings. Result is written to
backend/models/commit.json, which inference.py reads at startup.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "backend"))
from inference import SignRecognizer  # noqa: E402
from features import FRAME_DIM  # noqa: E402

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "..", "backend", "models")


def stream(rec, seq, rest):
    rec.reset()
    fired, t = [], 0.0
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
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--allow-idle", type=int, default=0,
                    help="max acceptable false fires on idle data (keep at 0)")
    args = ap.parse_args()

    labels = json.load(open(os.path.join(args.models, "labels.json")))
    none_files = glob.glob(os.path.join(args.data, "NONE", "*.npy"))
    if not none_files:
        sys.exit("no NONE recordings — record idle data first, it is what keeps the app quiet")
    rest = np.load(none_files[0])[:15]
    idle = [np.load(f) for f in none_files]

    test = []          # (gloss, sequence)
    for g in labels:
        if g == "NONE":
            continue
        for f in glob.glob(os.path.join(args.data, g, f"{args.holdout}_*.npy")):
            test.append((g, np.load(f)))
    if not test:
        sys.exit(f"no sequences for signer '{args.holdout}'")
    print(f"{len(test)} test sequences, {len(idle)} idle sequences, {len(labels)} classes\n")

    print(f"{'thresh':>7}{'consec':>8}{'idle':>7}{'acc':>8}{'silent':>8}")
    print("-" * 38)
    best = None
    for consec in (2, 1):
        for thresh in [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]:
            rec = SignRecognizer(args.models, thresh=thresh, consec=consec)
            fires = sum(len(stream(rec, s, rest)) for s in idle)
            if fires > args.allow_idle:
                print(f"{thresh:>7.2f}{consec:>8}{fires:>7}{'—':>8}{'—':>8}   (too noisy)")
                continue
            hit = silent = 0
            for g, seq in test:
                f = stream(rec, seq, rest)
                if not f:
                    silent += 1
                elif f[0] == g:
                    hit += 1
            acc, sil = hit / len(test), silent / len(test)
            print(f"{thresh:>7.2f}{consec:>8}{fires:>7}{acc:>8.3f}{sil:>8.3f}")
            # prefer accuracy; tie-break toward the higher (safer) threshold
            key = (round(acc, 3), thresh)
            if best is None or key > best[0]:
                best = (key, {"thresh": thresh, "consec": consec, "acc": acc,
                              "silent": sil, "idle_fires": fires})

    if best is None:
        sys.exit("\nEvery setting fired on idle data. Record more NONE, or retrain.")
    cfg = best[1]
    out = os.path.join(args.models, "commit.json")
    json.dump({"thresh": cfg["thresh"], "consec": cfg["consec"],
               "measured": {"live_accuracy": round(cfg["acc"], 3),
                            "silent_rate": round(cfg["silent"], 3),
                            "idle_false_fires": cfg["idle_fires"],
                            "classes": len(labels), "holdout": args.holdout}},
              open(out, "w"), indent=1)
    print(f"\nBest: thresh={cfg['thresh']:.2f} consec={cfg['consec']}  ->  "
          f"live accuracy {cfg['acc']:.3f}, silent {cfg['silent']:.3f}, idle false-fires {cfg['idle_fires']}")
    print(f"Wrote {out}  — restart the backend to use it.")


if __name__ == "__main__":
    main()
