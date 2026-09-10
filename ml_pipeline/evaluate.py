"""
Evaluate the exported model the way the demo will actually use it, then tell you
exactly which signs to re-record.

  python evaluate.py --holdout bhargav
  python evaluate.py --holdout bhargav --min 0.90     # stricter bar

Prints, and writes two files:
  backend/models/weak.json   per-class live accuracy + what each class is confused with
  ml_pipeline/redo.bat       double-click to re-record every weak sign, then retrain

Three numbers matter:
  [idle test]   candidates fired over your NONE recordings          -> must be 0
  [live-style]  first candidate == label, streamed through the real
                sliding-window commit logic                          -> want >= 0.90
  per-class     which signs are dragging the average down
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
REST = None          # real NONE frames used to pad sequences (set in main)


def stream(rec: SignRecognizer, seq: np.ndarray) -> list[str]:
    """Feed one sequence through the live recogniser, padded with real rest frames."""
    rec.reset()
    rest = REST if REST is not None else np.zeros((10, FRAME_DIM), np.float32)
    fired, t = [], 0.0
    for v in np.concatenate([rest, seq, rest]):
        t += 1 / 30
        r = rec.push(v, now=t)
        if r and r[0] == "candidate":
            fired.append(r[1])
    return fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, help="signer whose recordings to test on")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--models", default=None)
    ap.add_argument("--min", type=float, default=0.85, help="a class below this is 'weak'")
    ap.add_argument("--samples", type=int, default=25, help="samples to suggest per redo")
    args = ap.parse_args()

    rec = SignRecognizer(args.models) if args.models else SignRecognizer()
    labels = rec.labels
    L = {g: i for i, g in enumerate(labels)}

    # ---- idle false-fire test --------------------------------------------
    global REST
    none_files = glob.glob(os.path.join(args.data, "NONE", "*.npy"))
    if none_files:
        REST = np.load(none_files[0])[:15]
    fires = sum(len(stream(rec, np.load(f))) for f in none_files)
    print(f"[idle test] {len(none_files)} NONE sequences -> {fires} false candidates  "
          f"({'PASS' if fires == 0 else 'FAIL: raise THRESH/CONSEC or record more NONE'})")

    # ---- live-style accuracy + confusion ---------------------------------
    cm = np.zeros((len(labels), len(labels) + 1), int)   # last column = nothing fired
    n = hit = 0
    counts = {}
    for g in labels:
        if g == "NONE":
            continue
        files = glob.glob(os.path.join(args.data, g, f"{args.holdout}_*.npy"))
        counts[g] = len(files)
        for f in files:
            fired = stream(rec, np.load(f))
            n += 1
            if fired:
                cm[L[g], L[fired[0]]] += 1
                hit += int(fired[0] == g)
            else:
                cm[L[g], -1] += 1
    if n == 0:
        sys.exit(f"no sequences for signer '{args.holdout}' — check --holdout")
    print(f"[live-style] first candidate == label: {hit}/{n} = {hit / n:.3f}   "
          f"(nothing fired: {cm[:, -1].sum()})")

    # ---- per class -------------------------------------------------------
    rows, weak = [], []
    for g in labels:
        if g == "NONE" or counts.get(g, 0) == 0:
            continue
        total = cm[L[g]].sum()
        acc = cm[L[g], L[g]] / total if total else 0.0
        silent = cm[L[g], -1] / total if total else 0.0
        others = [(labels[j], int(cm[L[g], j])) for j in range(len(labels))
                  if j != L[g] and cm[L[g], j] > 0]
        others.sort(key=lambda kv: -kv[1])
        rows.append({"gloss": g, "samples": counts[g], "acc": round(float(acc), 3),
                     "silent": round(float(silent), 3),
                     "confused_with": [{"gloss": o, "n": c} for o, c in others[:2]]})
        if acc < args.min:
            weak.append(rows[-1])
    rows.sort(key=lambda r: r["acc"])

    print(f"\n{'sign':<18}{'acc':>7}{'silent':>8}   confused with")
    print("-" * 62)
    for r in rows:
        cw = ", ".join(f"{c['gloss']} x{c['n']}" for c in r["confused_with"]) or "-"
        flag = "  <-- WEAK" if r["acc"] < args.min else ""
        print(f"{r['gloss']:<18}{r['acc']:>7.2f}{r['silent']:>8.2f}   {cw}{flag}")

    # ---- what to do ------------------------------------------------------
    os.makedirs(MODELS, exist_ok=True)
    json.dump({"holdout": args.holdout, "threshold": args.min,
               "live_accuracy": round(hit / n, 3), "idle_false_fires": fires,
               "classes": rows}, open(os.path.join(MODELS, "weak.json"), "w"), indent=1)

    if not weak:
        print(f"\nNothing below {args.min:.2f}. Good to demo.")
        return

    print(f"\n{len(weak)} sign(s) below {args.min:.2f} — re-record these:\n")
    for r in weak:
        cw = r["confused_with"][0]["gloss"] if r["confused_with"] else None
        why = (f"looks like {cw} — make the MOTION clearly different"
               if cw else "often nothing fires — sign bigger, slower, both hands in frame")
        print(f"  {r['gloss']:<18} {r['acc']:.2f}   {why}")

    bat = os.path.join(ROOT, "redo.bat")
    with open(bat, "w") as f:
        f.write("@echo off\r\ncd /d %~dp0\r\n")
        f.write("echo Re-recording weak signs. Old samples are kept; new ones are added.\r\n")
        for r in weak:
            f.write(f"echo.\r\necho === {r['gloss']} (was {r['acc']:.2f}) ===\r\n")
            f.write(f"python collect_data.py --gloss {r['gloss']} --signer {args.holdout} "
                    f"--samples {args.samples} --auto\r\n")
        f.write("python train_mlp.py --epochs 40 --aug 4\r\n")
        f.write(f"python evaluate.py --holdout {args.holdout} --min {args.min}\r\n")
        f.write("pause\r\n")
    print(f"\nWrote {bat}\n  Double-click it, or run:  redo.bat")
    print("  It records only the weak signs, retrains, and re-checks. Restart the backend after.")

    # ---- confusion plot (optional) ---------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels) + 1)); ax.set_xticklabels(labels + ["(none)"], rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("first candidate fired"); ax.set_ylabel("true sign")
        out = os.path.join(MODELS, "confusion.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); print("saved", out)
    except Exception as e:
        print("(no confusion plot:", e, ")")


if __name__ == "__main__":
    main()
