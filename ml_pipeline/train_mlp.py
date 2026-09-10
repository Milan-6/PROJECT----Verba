"""
Train the SignBridge MLP on window features and export to ONNX.

  python train_mlp.py --holdout priya                 # honest: validate on a signer never trained on
  python train_mlp.py --holdout priya --epochs 60 --aug 6

Outputs (backend/models/):  model.onnx, labels.json, scaler.json, report.json

Why an MLP on window statistics (and not raw frames or an LSTM):
the thesis showed that with ~16 clips per class, raw keypoint sequences (8-12k numbers)
cannot be learned directly; it reduced them to limb descriptors first. We do the same
(mean/std/delta + velocity/acceleration stats -> 1322 numbers) and let a small MLP
separate the classes. This trains in seconds on CPU and runs in <5 ms at inference.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from features import window_features, augment, WINDOW_DIM, FRAME_DIM, WINDOW_T

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "..", "backend", "models")


class SignMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, n_out))

    def forward(self, x):
        return self.net(x)


def load_dataset(data_dir: str, min_per_class: int):
    seqs, labels, signers = [], [], []
    glosses = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    kept = []
    for g in glosses:
        files = sorted(glob.glob(os.path.join(data_dir, g, "*.npy")))
        if len(files) < min_per_class:
            print(f"[skip] {g}: only {len(files)} samples (<{min_per_class})"); continue
        kept.append(g)
        for f in files:
            a = np.load(f)
            if a.ndim != 2 or a.shape[1] != FRAME_DIM:
                print(f"[skip] bad shape {a.shape} in {f}"); continue
            seqs.append(a.astype(np.float32)); labels.append(g)
            signers.append(os.path.basename(f).rsplit("_", 1)[0])
    if "NONE" not in kept:
        print("[warn] no NONE class found — the live system WILL fire on idle hands. Record NONE data.")
    return seqs, labels, signers, kept


def build(seqs, labels, label_to_idx, n_aug: int, rng) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    rest_pool = [s for s, l in zip(seqs, labels) if l == "NONE"]   # real idle frames for padding
    for s, l in zip(seqs, labels):
        X.append(window_features(s)); y.append(label_to_idx[l])
        for _ in range(n_aug):
            X.append(window_features(augment(s, rng, rest_pool))); y.append(label_to_idx[l])
    return np.stack(X), np.array(y, dtype=np.int64)


def run_split(Xtr, ytr, Xva, yva, n_cls, epochs, seed, quiet=False):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd
    model = SignMLP(WINDOW_DIM, n_cls)
    counts = np.bincount(ytr, minlength=n_cls).astype(np.float32)
    w = torch.tensor(counts.sum() / (n_cls * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.tensor(Xtr_n), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva_n), torch.tensor(yva)
    best_f1, best_state, best_ep = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad(); loss = loss_fn(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(1)
        acc = (pred == yv).float().mean().item()
        f1 = macro_f1(yv.numpy(), pred.numpy(), n_cls)
        if f1 > best_f1:
            best_f1, best_ep = f1, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if not quiet and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}  loss {loss.item():.3f}  val acc {acc:.3f}  macroF1 {f1:.3f}")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(Xv).argmax(1).numpy()
    return model, mu, sd, (pred == yva).mean(), best_f1, best_ep, pred


def macro_f1(y, p, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--holdout", help="signer name to hold out for honest validation")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--aug", type=int, default=4, help="augmented copies per sample")
    ap.add_argument("--min-per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    seqs, labels, signers, glosses = load_dataset(args.data, args.min_per_class)
    if not seqs:
        sys.exit("no data found in " + args.data)
    label_to_idx = {g: i for i, g in enumerate(glosses)}
    n_cls = len(glosses)
    print(f"{len(seqs)} sequences, {n_cls} classes, signers={sorted(set(signers))}")

    report = {"classes": glosses, "n_sequences": len(seqs)}

    # ---- 1) random split (optimistic; reported only for comparison) --------
    idx = rng.permutation(len(seqs)); cut = int(len(idx) * 0.8)
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr = build([seqs[i] for i in tr], [labels[i] for i in tr], label_to_idx, args.aug, rng)
    Xva, yva = build([seqs[i] for i in va], [labels[i] for i in va], label_to_idx, 0, rng)
    print("\n[random split]  (same hands in train and val — optimistic)")
    _, _, _, acc_r, f1_r, _, _ = run_split(Xtr, ytr, Xva, yva, n_cls, args.epochs, args.seed, quiet=True)
    print(f"  random-split acc {acc_r:.3f}  macroF1 {f1_r:.3f}")
    report["random_split"] = {"acc": float(acc_r), "macro_f1": float(f1_r)}

    # ---- 2) signer-held-out split (honest) ----------------------------------
    if args.holdout:
        tr = [i for i, s in enumerate(signers) if s != args.holdout]
        va = [i for i, s in enumerate(signers) if s == args.holdout]
        if not va:
            sys.exit(f"holdout signer '{args.holdout}' has no data; signers = {sorted(set(signers))}")
        Xtr, ytr = build([seqs[i] for i in tr], [labels[i] for i in tr], label_to_idx, args.aug, rng)
        Xva, yva = build([seqs[i] for i in va], [labels[i] for i in va], label_to_idx, 0, rng)
        print(f"\n[held-out signer = {args.holdout}]  (honest number — this is what the stage will feel like)")
        _, _, _, acc_h, f1_h, ep_h, pred = run_split(Xtr, ytr, Xva, yva, n_cls, args.epochs, args.seed)
        print(f"  held-out acc {acc_h:.3f}  macroF1 {f1_h:.3f}  (best epoch {ep_h})")
        report["held_out"] = {"signer": args.holdout, "acc": float(acc_h), "macro_f1": float(f1_h)}
        per = {}
        for c, g in enumerate(glosses):
            m = yva == c
            if m.any():
                per[g] = float((pred[m] == c).mean())
        report["held_out"]["per_class_acc"] = per
        worst = sorted(per.items(), key=lambda kv: kv[1])[:5]
        print("  weakest classes:", ", ".join(f"{g} {a:.2f}" for g, a in worst))
        if acc_h < 0.9:
            print("  >> below the 90% target: record more samples for the weakest classes, "
                  "check both hands are tracked, and re-run.")

    # ---- 3) final model on ALL data -> ONNX ---------------------------------
    print("\n[final] training on all signers for deployment")
    Xall, yall = build(seqs, labels, label_to_idx, args.aug, rng)
    model, mu, sd, train_acc, train_f1, _, _ = run_split(Xall, yall, Xall[:64], yall[:64], n_cls, args.epochs, args.seed, quiet=True)
    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "model.onnx")
    dummy = torch.zeros(1, WINDOW_DIM)
    torch.onnx.export(model, dummy, onnx_path, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)

    # 1. Labels and class mappings
    json.dump(glosses, open(os.path.join(args.out, "labels.json"), "w"), indent=1)
    class_to_idx = {g: i for i, g in enumerate(glosses)}
    idx_to_class = {i: g for i, g in enumerate(glosses)}
    json.dump(class_to_idx, open(os.path.join(args.out, "class_to_index.json"), "w"), indent=2)
    json.dump(idx_to_class, open(os.path.join(args.out, "index_to_class.json"), "w"), indent=2)

    # 2. Scaler
    json.dump({"mean": mu.tolist(), "std": sd.tolist()}, open(os.path.join(args.out, "scaler.json"), "w"))

    # 3. Preprocessing configuration
    preprocessing_config = {
        "sequence_length": WINDOW_T,
        "frame_feature_dim": FRAME_DIM,
        "window_feature_dim": WINDOW_DIM,
        "normalization": "body_scale_and_mean_std",
        "fps_resample": 30,
        "feature_composition": {
            "presence_flags": 2,
            "body_normalised_landmarks": 140,
            "limb_vectors": 128,
            "window_statistics": "mean, std, delta, vel_mean, vel_std, acc_mean, acc_std",
        },
    }
    json.dump(preprocessing_config, open(os.path.join(args.out, "preprocessing.json"), "w"), indent=2)

    # 4. Training configuration
    training_config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss(label_smoothing=0.05)",
        "augmentation_factor": args.aug,
        "num_classes": n_cls,
        "num_sequences": len(seqs),
        "classes": glosses,
        "device": "cpu",
    }
    json.dump(training_config, open(os.path.join(args.out, "training_config.json"), "w"), indent=2)

    # 5. Evaluation metrics
    eval_results = {
        "train_accuracy": float(acc_r),
        "validation_accuracy": float(acc_h if args.holdout else acc_r),
        "macro_f1": float(f1_h if args.holdout else f1_r),
        "random_split": report.get("random_split", {}),
        "held_out": report.get("held_out", {}),
    }
    json.dump(eval_results, open(os.path.join(args.out, "evaluation.json"), "w"), indent=2)

    # 6. Commit configuration
    commit_config = {
        "thresh": 0.65,
        "consec": 2,
        "stride": 3,
        "smooth": 2,
        "refractory_s": 0.4,
        "classes": n_cls,
    }
    json.dump(commit_config, open(os.path.join(args.out, "commit.json"), "w"), indent=2)

    # 7. Report
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)

    # verify ONNX == PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    xs = ((Xall[:10] - mu) / sd).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.tensor(xs)).numpy()
    out = sess.run(None, {"x": xs})[0]
    print(f"[export] {onnx_path}  max |onnx - torch| = {np.abs(out - ref).max():.2e}")
    t0 = time.time(); [sess.run(None, {"x": xs[:1]}) for _ in range(100)]
    print(f"[export] inference {(time.time() - t0) * 10:.2f} ms / sample on CPU")
    print(f"[export] Saved all artifacts to {args.out}")


if __name__ == "__main__":
    main()
