"""
SignRecognizer: ring buffer -> window features -> ONNX MLP -> smoothed candidate.
One instance per WebSocket connection.

Commit logic (protects the live demo):
  * classify every STRIDE frames over the last WINDOW_T frames
  * average the last 2 probability vectors
  * a gloss becomes a *candidate* only if it is not NONE, its averaged prob > THRESH,
    and it has been the arg-max for 2 consecutive windows (~0.15 s at 20 fps)
  * after emitting, a REFRACTORY period prevents the same sign firing twice
  * nothing is ever spoken from here — the client asks the signer to confirm
"""
from __future__ import annotations
import json, os, sys, time
from collections import deque
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml_pipeline"))
from features import window_features, WINDOW_T, FRAME_DIM  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# Defaults. models/commit.json (written by ml_pipeline/calibrate.py) overrides them:
# the right threshold depends on how many classes you trained, so it is measured, not guessed.
STRIDE, THRESH, CONSEC, REFRACTORY_S, SMOOTH = 3, 0.75, 2, 0.4, 2


class SignRecognizer:
    def __init__(self, model_dir: str = MODELS, thresh: float | None = None,
                 consec: int | None = None, smooth: int | None = None, stride: int | None = None):
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), providers=["CPUExecutionProvider"])
        self.labels: list[str] = json.load(open(os.path.join(model_dir, "labels.json")))
        sc = json.load(open(os.path.join(model_dir, "scaler.json")))
        self.mu, self.sd = np.array(sc["mean"], np.float32), np.array(sc["std"], np.float32)
        self.none_idx = self.labels.index("NONE") if "NONE" in self.labels else -1
        cfg = {}
        cfg_path = os.path.join(model_dir, "commit.json")
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path))
            except Exception:
                cfg = {}
        self.thresh = THRESH if thresh is None else thresh
        self.consec = CONSEC if consec is None else consec
        self.smooth = SMOOTH if smooth is None else smooth
        self.stride = STRIDE if stride is None else stride
        if thresh is None: self.thresh = float(cfg.get("thresh", self.thresh))
        if consec is None: self.consec = int(cfg.get("consec", self.consec))
        if smooth is None: self.smooth = int(cfg.get("smooth", self.smooth))
        if stride is None: self.stride = int(cfg.get("stride", self.stride))
        self.reset()

    def reset(self):
        self.frames: deque = deque(maxlen=WINDOW_T)
        self.probs: deque = deque(maxlen=self.smooth)
        self.n_since = 0
        self.streak_label, self.streak = -1, 0
        self.last_emit_t = 0.0

    def _predict(self) -> np.ndarray:
        x = window_features(np.stack(self.frames))
        x = ((x - self.mu) / self.sd).astype(np.float32)[None]
        logits = self.sess.run(None, {"x": x})[0][0]
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def push(self, vec, now: float | None = None):
        """vec: 270 floats. Returns None, ("idle",), or ("candidate", gloss, conf)."""
        v = np.asarray(vec, np.float32)
        if v.shape != (FRAME_DIM,):
            raise ValueError(f"expected {FRAME_DIM} floats, got {v.shape}")
        now = time.time() if now is None else now
        self.frames.append(v); self.n_since += 1
        if len(self.frames) < WINDOW_T or self.n_since < self.stride:
            return None
        self.n_since = 0
        # hands absent for the whole window -> idle, cheap path
        if v[0] == 0 and v[1] == 0 and all(f[0] == 0 and f[1] == 0 for f in self.frames):
            self.probs.clear(); self.streak = 0
            return ("idle",)
        self.probs.append(self._predict())
        avg = np.mean(self.probs, axis=0)
        top = int(avg.argmax()); conf = float(avg[top])
        if top == self.none_idx or conf < self.thresh:
            self.streak_label, self.streak = top, 0
            return ("idle",)
        self.streak = self.streak + 1 if top == self.streak_label else 1
        self.streak_label = top
        if self.streak >= self.consec and now - self.last_emit_t > REFRACTORY_S:
            self.last_emit_t = now; self.streak = 0; self.probs.clear()
            return ("candidate", self.labels[top], round(conf, 3))
        return None
