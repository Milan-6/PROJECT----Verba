"""
SignBridge feature pipeline — single source of truth for Python (training + backend).
frontend/src/lib/features.ts is an exact mirror; keep the two in sync.

Design follows the IIT-Madras ISL thesis (Sridhar, 2019, "Sign Language Translation"):
  * skeletal keypoints -> LIMB position vectors  P_l = j_end - j_start
  * limb lengths normalised by a body-relative scale (thesis: eye distance;
    we default to shoulder width because it is far more stable at 1-2 m, and
    fall back to eye distance if shoulders are not visible)
  * limb VELOCITY  V_t = P_t - P_{t-1}  and ACCELERATION  A_t = V_t - V_{t-1}
  * per-limb statistics over a short window instead of raw frames, because we
    have few samples per class (thesis had ~16/class; we have ~100)

We add two things the thesis did not have:
  * body-normalised absolute landmark positions (WHERE the sign happens —
    head / chest / waist — which ISL grammar depends on)
  * an explicit NONE (rest) class handled downstream

Per-frame vector layout (FRAME_DIM = 270):
  [0:2]      presence flags   left_present, right_present            (2)
  [2:142]    body-normalised  L hand 21x3, R hand 21x3, pose 7x2     (140)
  [142:270]  limb vectors     arms 4x2, fingers 40x3                 (128)

Window vector (WINDOW_DIM = 1322) for T frames:
  mean/std/delta of the 270 frame features                          (810)
  mean/std of limb velocity (128 dims each)                         (256)
  mean/std of limb acceleration                                     (256)
"""
from __future__ import annotations
import numpy as np

# ---- MediaPipe index maps -------------------------------------------------
POSE_NOSE, POSE_LEYE, POSE_REYE = 0, 2, 5
POSE_LSHO, POSE_RSHO, POSE_LELB, POSE_RELB, POSE_LWRI, POSE_RWRI = 11, 12, 13, 14, 15, 16
POSE_KEEP = [POSE_NOSE, POSE_LSHO, POSE_RSHO, POSE_LELB, POSE_RELB, POSE_LWRI, POSE_RWRI]

# finger limbs: (start, end) pairs, 4 per finger, thumb..pinky (thesis Fig 5.8)
FINGER_LIMBS = [(0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
# arm limbs on pose: shoulder->elbow, elbow->wrist, both sides (thesis Fig 5.7)
ARM_LIMBS = [(POSE_LSHO, POSE_LELB), (POSE_LELB, POSE_LWRI),
             (POSE_RSHO, POSE_RELB), (POSE_RELB, POSE_RWRI)]

FRAME_DIM = 270
LIMB_DIM = 128
RECORD_T = 30   # frames saved per training sample (collect_data / preprocess_video)
WINDOW_T = 24   # frames the LIVE recogniser classifies over. Shorter = faster response.
                # Safe to differ from RECORD_T because window_features() is length-agnostic
                # (mean/std/delta) and training augments with time-stretched windows.
WINDOW_DIM = FRAME_DIM * 3 + LIMB_DIM * 4  # 1322
_EPS = 1e-6


def assign_hands(hands: list[np.ndarray], pose: np.ndarray | None):
    """Return (left, right) hand arrays (21,3) or None, assigned by proximity to the
    pose wrists. Robust to mirroring, unlike MediaPipe's handedness label."""
    left = right = None
    if not hands:
        return None, None
    if pose is None:
        # no pose: fall back to image x (person faces camera => their right hand is on image-left)
        hs = sorted(hands, key=lambda h: h[0, 0])
        if len(hs) == 1:
            return (None, hs[0]) if hs[0][0, 0] < 0.5 else (hs[0], None)
        return hs[1], hs[0]
    lw, rw = pose[POSE_LWRI, :2], pose[POSE_RWRI, :2]
    scored = []
    for h in hands:
        w = h[0, :2]
        scored.append((np.linalg.norm(w - lw), np.linalg.norm(w - rw), h))
    if len(scored) == 1:
        dl, dr, h = scored[0]
        return (h, None) if dl <= dr else (None, h)
    # two hands: choose assignment minimising total distance
    (dl0, dr0, h0), (dl1, dr1, h1) = scored[:2]
    if dl0 + dr1 <= dr0 + dl1:
        return h0, h1
    return h1, h0


def body_scale(pose: np.ndarray | None) -> float:
    """Thesis normalises by eye distance; shoulder width is steadier at 1-2 m."""
    if pose is None:
        return 1.0
    sw = np.linalg.norm(pose[POSE_LSHO, :2] - pose[POSE_RSHO, :2])
    if sw > 0.02:
        return float(sw)
    ew = np.linalg.norm(pose[POSE_LEYE, :2] - pose[POSE_REYE, :2])
    return float(ew * 4.0) if ew > 0.005 else 1.0  # eye dist ≈ shoulder/4


def frame_vector(left: np.ndarray | None, right: np.ndarray | None,
                 pose: np.ndarray | None) -> np.ndarray:
    """left/right: (21,3) normalised image coords or None. pose: (33,3+) or None.
    Returns float32 array of FRAME_DIM."""
    v = np.zeros(FRAME_DIM, dtype=np.float32)
    v[0] = 1.0 if left is not None else 0.0
    v[1] = 1.0 if right is not None else 0.0

    if pose is not None:
        c = (pose[POSE_LSHO, :2] + pose[POSE_RSHO, :2]) / 2.0
    else:
        c = np.array([0.5, 0.5], dtype=np.float32)
    s = body_scale(pose)

    # --- [2:142] body-normalised positions ---------------------------------
    o = 2
    for h in (left, right):
        if h is not None:
            hn = h[:, :3].copy()
            hn[:, :2] = (hn[:, :2] - c) / s
            hn[:, 2] = hn[:, 2] / s
            v[o:o + 63] = hn.reshape(-1)
        o += 63
    if pose is not None:
        pn = (pose[POSE_KEEP, :2] - c) / s
        v[o:o + 14] = pn.reshape(-1)
    o += 14
    assert o == 142

    # --- [142:270] limb position vectors (thesis P_l = j_end - j_start) -----
    if pose is not None:
        for (a, b) in ARM_LIMBS:
            v[o:o + 2] = (pose[b, :2] - pose[a, :2]) / s
            o += 2
    else:
        o += 8
    for h in (left, right):
        if h is not None:
            # finger limbs normalised by hand size so handshape is distance-free
            hs = np.linalg.norm(h[9, :3] - h[0, :3]) + _EPS
            for (a, b) in FINGER_LIMBS:
                v[o:o + 3] = (h[b, :3] - h[a, :3]) / hs
                o += 3
        else:
            o += 60
    assert o == FRAME_DIM
    return v


def window_features(frames: np.ndarray) -> np.ndarray:
    """frames: (T, FRAME_DIM) -> (WINDOW_DIM,). T may differ from WINDOW_T."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != FRAME_DIM:
        raise ValueError(f"expected (T,{FRAME_DIM}), got {frames.shape}")
    if frames.shape[0] < 3:
        frames = np.concatenate([frames] + [frames[-1:]] * (3 - frames.shape[0]))
    mean = frames.mean(0)
    std = frames.std(0)
    delta = frames[-1] - frames[0]
    limbs = frames[:, 142:270]
    vel = np.diff(limbs, axis=0)          # V_t = P_t - P_{t-1}
    acc = np.diff(vel, axis=0)            # A_t = V_t - V_{t-1}
    out = np.concatenate([mean, std, delta,
                          vel.mean(0), vel.std(0),
                          acc.mean(0), acc.std(0)]).astype(np.float32)
    assert out.shape[0] == WINDOW_DIM
    return out


# ---- augmentation (training only) ----------------------------------------
def mirror(frames: np.ndarray) -> np.ndarray:
    """Swap left/right hands and negate x so left-handed signers are covered."""
    f = frames.copy()
    f[:, [0, 1]] = f[:, [1, 0]]
    L, R = f[:, 2:65].copy(), f[:, 65:128].copy()
    f[:, 2:65], f[:, 65:128] = R, L
    f[:, 2:128:3] *= -1                      # x of hand landmarks
    pose = f[:, 128:142].reshape(-1, 7, 2)   # nose, Lsho, Rsho, Lelb, Relb, Lwri, Rwri
    pose = pose[:, [0, 2, 1, 4, 3, 6, 5], :]
    pose[:, :, 0] *= -1
    f[:, 128:142] = pose.reshape(-1, 14)
    arms = f[:, 142:150].reshape(-1, 4, 2)[:, [2, 3, 0, 1], :]
    arms[:, :, 0] *= -1
    f[:, 142:150] = arms.reshape(-1, 8)
    fl, fr = f[:, 150:210].copy(), f[:, 210:270].copy()
    f[:, 150:210], f[:, 210:270] = fr, fl
    f[:, 150:270:3] *= -1
    return f


def jitter(frames: np.ndarray, sigma: float = 0.01, rng=np.random) -> np.ndarray:
    f = frames.copy()
    f[:, 2:] += rng.normal(0, sigma, f[:, 2:].shape).astype(np.float32)
    return f


def time_stretch(frames: np.ndarray, factor: float) -> np.ndarray:
    """Resample the sequence to round(T*factor) frames (±20 % speed)."""
    T = frames.shape[0]
    n = max(3, int(round(T * factor)))
    idx = np.linspace(0, T - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (idx - lo)[:, None]
    return (frames[lo] * (1 - w) + frames[hi] * w).astype(np.float32)


def frame_dropout(frames: np.ndarray, p: float = 0.05, rng=np.random) -> np.ndarray:
    keep = rng.random(frames.shape[0]) > p
    keep[0] = keep[-1] = True
    return frames[keep]


def shift_pad(frames: np.ndarray, rng=np.random, max_frac: float = 0.4,
              rest_pool: list[np.ndarray] | None = None) -> np.ndarray:
    """Live recognition sees SLIDING windows, so a sign often occupies only part of the
    window with rest before/after. Simulate that: pad with rest frames and crop back to
    the same length. Rest frames come from a real NONE sequence when rest_pool is given
    (what the camera actually sees between signs), else the sequence's own first/last frame."""
    T = frames.shape[0]
    k = int(rng.integers(0, max(1, int(T * max_frac)) + 1))
    if k == 0:
        return frames
    if rest_pool:
        src = rest_pool[int(rng.integers(len(rest_pool)))]
        j = int(rng.integers(0, max(1, src.shape[0] - k + 1)))
        pad = src[j:j + k]
        if pad.shape[0] < k:
            pad = np.concatenate([pad, np.repeat(pad[-1:], k - pad.shape[0], 0)])
        pad_before, pad_after = pad, pad
    else:
        pad_before, pad_after = np.repeat(frames[:1], k, 0), np.repeat(frames[-1:], k, 0)
    if rng.random() < 0.5:
        padded = np.concatenate([pad_before, frames])[:T]
    else:
        padded = np.concatenate([frames, pad_after])[k:]
    return padded.astype(np.float32)


def augment(frames: np.ndarray, rng=np.random, rest_pool: list[np.ndarray] | None = None) -> np.ndarray:
    f = frames
    if rng.random() < 0.5:
        f = mirror(f)
    f = time_stretch(f, rng.uniform(0.7, 1.15))   # 30 frames -> 21..34, covers WINDOW_T=22
    f = frame_dropout(f, 0.05, rng)
    if rng.random() < 0.6:
        f = shift_pad(f, rng, rest_pool=rest_pool)
    f = jitter(f, 0.01, rng)
    return f
