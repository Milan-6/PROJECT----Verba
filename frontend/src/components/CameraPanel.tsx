/**
 * CameraPanel — drop-in replacement for the camera section in main.tsx.
 *
 *   <CameraPanel active={cameraOn} onVector={v => socket.sendFrame(v)} paused={pinFocused} />
 *
 * Owns: getUserMedia (via useCamera), MediaPipe landmarks (via useLandmarks), the mirrored
 * <video>, the skeleton <canvas> overlay, the camera picker, and the tracking-quality chip.
 * Emits only the 270-float vector. Never sends pixels anywhere.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useCamera } from '../hooks/useCamera';
import { useLandmarks, LandmarkFrame, Quality, ModelStatus } from '../hooks/useLandmarks';
import { Pt } from '../lib/features';

const HAND_CONN: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const QUALITY_TEXT: Record<Quality, string> = {
  good: 'Tracking: both hands',
  'one-hand': 'Tracking: one hand',
  'no-hands': 'Show your hands',
  'too-far': 'Come closer',
};

// Kept chromatic on purpose: these encode tracking state a Deaf user acts on.
const QUALITY_COLOR: Record<Quality, string> = {
  good: 'var(--ok, #2e7d32)',
  'one-hand': 'var(--warn, #d97706)',
  'no-hands': 'var(--ink-faint, #A39C90)',
  'too-far': 'var(--warn, #d97706)',
};

interface Props {
  active: boolean;                       // camera on/off (controlled by the parent's button)
  paused?: boolean;                      // e.g. bank PIN field focused -> stop the stream
  onVector: (v: Float32Array) => void;   // ~20 fps
  onQuality?: (q: Quality) => void;
  onModelStatusChange?: (status: ModelStatus) => void;
}

export default function CameraPanel({ active, paused = false, onVector, onQuality, onModelStatusChange }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const cam = useCamera(video);
  const [quality, setQuality] = useState<Quality>('no-hands');
  const shouldRun = active && !paused;

  // start/stop follows the parent's intent; deps are stable so this cannot loop
  useEffect(() => {
    if (shouldRun) cam.start(); else cam.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shouldRun]);

  const draw = useCallback((f: LandmarkFrame) => {
    const c = canvas.current, v = video.current;
    if (!c || !v) return;
    if (c.width !== v.videoWidth || c.height !== v.videoHeight) {
      c.width = v.videoWidth;
      c.height = v.videoHeight;
    }
    const g = c.getContext('2d')!;
    g.clearRect(0, 0, c.width, c.height);
    const col = f.quality === 'good' ? '#2e7d32' : f.quality === 'no-hands' ? '#A39C90' : '#d97706';
    const P = (p: Pt) => [p.x * c.width, p.y * c.height] as const;
    for (const h of [f.left, f.right]) {
      if (!h) continue;
      g.strokeStyle = col;
      g.lineWidth = 2;
      g.fillStyle = col;
      for (const [a, b] of HAND_CONN) {
        g.beginPath();
        g.moveTo(...P(h[a]));
        g.lineTo(...P(h[b]));
        g.stroke();
      }
      for (const p of h) {
        g.beginPath();
        g.arc(...P(p), 3, 0, Math.PI * 2);
        g.fill();
      }
    }
    if (f.pose) {
      g.strokeStyle = 'rgba(255,255,255,.85)';
      g.lineWidth = 2;
      for (const [a, b] of [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]] as [number, number][]) {
        g.beginPath();
        g.moveTo(...P(f.pose[a]));
        g.lineTo(...P(f.pose[b]));
        g.stroke();
      }
    }
  }, []);

  const onFrame = useCallback((f: LandmarkFrame) => {
    draw(f);
    if (f.quality !== quality) {
      setQuality(f.quality);
      onQuality?.(f.quality);
    }
    onVector(f.vector);
  }, [draw, onVector, onQuality, quality]);

  const { modelStatus } = useLandmarks(video, cam.status === 'on' && shouldRun, onFrame, 20);
  const isLive = cam.status === 'on' && shouldRun;

  useEffect(() => {
    onModelStatusChange?.(modelStatus);
  }, [modelStatus, onModelStatusChange]);

  return (
    <div className="camera-panel-wrapper" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-3)' }}>
      <div className="camera-controls-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-2)' }}>
          <label
            htmlFor="cam"
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              color: 'var(--ink-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            Camera
          </label>
          <select
            id="cam"
            className="camera-picker-select"
            value={cam.deviceId}
            onChange={e => cam.switchTo(e.target.value)}
            disabled={cam.status === 'starting'}
          >
            {cam.devices.length === 0 && <option value="">Default camera</option>}
            {cam.devices.map((d, i) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        </div>
        <span className="quality-chip" style={{ color: QUALITY_COLOR[quality] }}>
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              backgroundColor: isLive && quality === 'good' ? 'var(--ok)' : isLive ? 'var(--warn)' : 'var(--ink-faint)',
              display: 'inline-block',
            }}
          />
          {modelStatus === 'loading'
            ? 'Loading model…'
            : modelStatus === 'error'
            ? 'Model error'
            : isLive
            ? QUALITY_TEXT[quality]
            : cam.status === 'starting'
            ? 'Starting…'
            : 'Camera off'}
        </span>
      </div>

      <div className={`camera-stage-frame ${isLive ? 'listening' : ''}`}>
        {/* mirrored so signing feels natural; the canvas is mirrored with it so overlays line up */}
        <video ref={video} playsInline muted />
        <canvas ref={canvas} />
        {paused && <div className="camera-overlay">Camera paused while entering PIN</div>}
        {cam.status === 'error' && (
          <div className="camera-overlay" style={{ flexDirection: 'column', gap: 'var(--s-2)', padding: 'var(--s-4)' }}>
            <span>{cam.error}</span>
            <button
              className="btn-outline"
              onClick={() => cam.start()}
              style={{ background: '#ffffff', color: '#171512' }}
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
