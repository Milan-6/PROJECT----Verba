"""
Shared MediaPipe Tasks wrapper: image/frame -> (left_hand, right_hand, pose) arrays.
Downloads the two .task model files on first use.
"""
from __future__ import annotations
import os, time, urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from features import assign_hands

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def _ensure(url: str, name: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"[mp] downloading {name} ...")
        urllib.request.urlretrieve(url, path)
    return path


class Extractor:
    """Use one Extractor per video stream. mode='video' keeps temporal tracking."""

    def __init__(self, mode: str = "video"):
        rm = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure(HAND_URL, "hand_landmarker.task")),
            running_mode=rm, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure(POSE_URL, "pose_landmarker_lite.task")),
            running_mode=rm, num_poses=1))
        self.mode = mode
        self._t0 = time.time()
        self._last_ts = -1

    def __call__(self, bgr: np.ndarray, ts_ms: int | None = None):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            if ts_ms is None:
                ts_ms = int((time.time() - self._t0) * 1000)
            ts_ms = max(ts_ms, self._last_ts + 1)   # timestamps must increase
            self._last_ts = ts_ms
            hr = self.hands.detect_for_video(img, ts_ms)
            pr = self.pose.detect_for_video(img, ts_ms)
        else:
            hr = self.hands.detect(img)
            pr = self.pose.detect(img)
        hands = [np.array([[l.x, l.y, l.z] for l in h], dtype=np.float32) for h in hr.hand_landmarks]
        pose = None
        if pr.pose_landmarks:
            pose = np.array([[l.x, l.y, l.z] for l in pr.pose_landmarks[0]], dtype=np.float32)
        left, right = assign_hands(hands, pose)
        return left, right, pose, hands

    def close(self):
        self.hands.close(); self.pose.close()


HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]


def draw(bgr: np.ndarray, left, right, pose, mirror: bool = False):
    """Draw on a display frame. If the display frame is a mirrored copy, pass mirror=True
    (landmarks were extracted from the UN-mirrored frame, exactly like the browser does)."""
    h, w = bgr.shape[:2]
    X = (lambda x: (1 - x) * w) if mirror else (lambda x: x * w)
    for hand, col in ((left, (255, 160, 0)), (right, (0, 200, 255))):
        if hand is None:
            continue
        pts = [(int(X(p[0])), int(p[1] * h)) for p in hand]
        for a, b in HAND_CONN:
            cv2.line(bgr, pts[a], pts[b], col, 2)
        for p in pts:
            cv2.circle(bgr, p, 3, col, -1)
    if pose is not None:
        for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16)):
            pa, pb = pose[a], pose[b]
            cv2.line(bgr, (int(X(pa[0])), int(pa[1] * h)), (int(X(pb[0])), int(pb[1] * h)), (0, 255, 120), 2)
    return bgr
