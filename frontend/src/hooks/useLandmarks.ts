/**
 * useLandmarks — MediaPipe Tasks (WASM, in-browser) -> 270-float frame vector at ~20 fps.
 * Raw video never leaves the browser; only the vector is handed to onVector().
 *
 * Model files are loaded from the official CDN once and cached by the browser. Show the
 * `modelStatus` in the UI: first load can take 5-20 s on venue Wi-Fi — that is the
 * "screen keeps loading" symptom, NOT a camera bug. Copy the two .task files into
 * /public/models and change the URLs below to make it fully offline.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { assignHands, frameVector, Pt } from '../lib/features';

// WASM is bundled in /public/mediapipe/wasm (copied from node_modules) so it works offline.
// Model files: run `npm run models` once to download them into /public/mediapipe; until then the CDN is used.
const WASM_LOCAL = '/mediapipe/wasm';
const WASM_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODELS = {
  hand: ['/mediapipe/hand_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'],
  pose: ['/mediapipe/pose_landmarker_lite.task', 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'],
};
async function pick(urls: string[]): Promise<string> {
  for (const u of urls) { try { const r = await fetch(u, { method: 'HEAD' }); if (r.ok) return u; } catch { /* try next */ } }
  return urls[urls.length - 1];
}

export type Quality = 'no-hands' | 'one-hand' | 'good' | 'too-far';
export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface LandmarkFrame { vector: Float32Array; left: Pt[] | null; right: Pt[] | null; pose: Pt[] | null; quality: Quality; }

export function useLandmarks(videoRef: React.RefObject<HTMLVideoElement>, active: boolean,
                             onFrame: (f: LandmarkFrame) => void, fps = 20) {
  const [modelStatus, setModelStatus] = useState<ModelStatus>('idle');
  const hand = useRef<HandLandmarker | null>(null);
  const pose = useRef<PoseLandmarker | null>(null);
  const raf = useRef(0);
  const lastT = useRef(0);
  const cb = useRef(onFrame);
  cb.current = onFrame;

  // load models once
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setModelStatus('loading');
        const wasm = (await fetch(WASM_LOCAL + '/vision_wasm_internal.js', { method: 'HEAD' }).then(r => r.ok).catch(() => false)) ? WASM_LOCAL : WASM_CDN;
        const files = await FilesetResolver.forVisionTasks(wasm);
        const [handUrl, poseUrl] = await Promise.all([pick(MODELS.hand), pick(MODELS.pose)]);
        const make = async (delegate: 'GPU' | 'CPU') => Promise.all([
          HandLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: handUrl, delegate }, runningMode: 'VIDEO', numHands: 2, minHandDetectionConfidence: 0.5, minTrackingConfidence: 0.5 }),
          PoseLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: poseUrl, delegate }, runningMode: 'VIDEO', numPoses: 1 }),
        ]);
        // GPU delegate fails on some laptops/WebGL blocklists -> fall back to CPU instead of dying
        const [h, p] = await make('GPU').catch(() => make('CPU'));
        if (cancelled) { h.close(); p.close(); return; }
        hand.current = h; pose.current = p;
        setModelStatus('ready');
      } catch (e) {
        console.error('MediaPipe load failed', e);
        setModelStatus('error');
      }
    })();
    return () => { cancelled = true; hand.current?.close(); pose.current?.close(); hand.current = null; pose.current = null; };
  }, []);

  const tick = useCallback(() => {
    raf.current = requestAnimationFrame(tick);
    const el = videoRef.current;
    if (!el || !hand.current || !pose.current || el.readyState < 2 || el.videoWidth === 0) return;
    const now = performance.now();
    if (now - lastT.current < 1000 / fps) return;
    lastT.current = now;
    const ts = Math.round(now);
    const hr = hand.current.detectForVideo(el, ts);
    const pr = pose.current.detectForVideo(el, ts);
    const hands = (hr.landmarks ?? []).map(h => h.map(l => ({ x: l.x, y: l.y, z: l.z })));
    const ps = pr.landmarks?.[0] ? pr.landmarks[0].map(l => ({ x: l.x, y: l.y, z: l.z })) : null;
    const { left, right } = assignHands(hands, ps);
    const vector = frameVector(left, right, ps);
    const n = (left ? 1 : 0) + (right ? 1 : 0);
    let quality: Quality = n === 2 ? 'good' : n === 1 ? 'one-hand' : 'no-hands';
    if (n && ps) {
      const sw = Math.hypot(ps[11].x - ps[12].x, ps[11].y - ps[12].y);
      if (sw < 0.12) quality = 'too-far';
    }
    cb.current({ vector, left, right, pose: ps, quality });
  }, [videoRef, fps]);

  useEffect(() => {
    if (!active || modelStatus !== 'ready') return;
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [active, modelStatus, tick]);

  return { modelStatus };
}
