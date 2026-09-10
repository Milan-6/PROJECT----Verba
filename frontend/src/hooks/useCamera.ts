/**
 * useCamera — loop-free webcam control.
 *
 * Why the old code froze after one second: startCamera called setSelectedCamera() inside a
 * useCallback that DEPENDED on selectedCamera. Each start changed the dependency, which
 * rebuilt startCamera, which re-triggered the effect/cleanup, which stopped and restarted
 * the stream — forever. This hook keeps the chosen device in a ref (no dependency churn),
 * never auto-switches devices, and always releases the previous stream before opening a new one.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraStatus = 'off' | 'starting' | 'on' | 'error';

export function useCamera(videoRef: React.RefObject<HTMLVideoElement>) {
  const [status, setStatus] = useState<CameraStatus>('off');
  const [error, setError] = useState('');
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState('');
  const chosen = useRef('');          // current device — a ref so callbacks never depend on it
  const stream = useRef<MediaStream | null>(null);
  const starting = useRef(false);     // guards against double-start from StrictMode / fast clicks

  const stop = useCallback(() => {
    stream.current?.getTracks().forEach(t => t.stop());
    stream.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus('off');
  }, [videoRef]);

  const start = useCallback(async (requestedId?: string) => {
    if (starting.current) return;
    starting.current = true;
    setError('');
    setStatus('starting');
    try {
      stop();
      const id = requestedId ?? chosen.current;
      const video: MediaTrackConstraints = id
        ? { deviceId: { exact: id }, width: { ideal: 960 }, height: { ideal: 540 } }
        : { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 540 } };
      let s: MediaStream;
      try {
        s = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      } catch (e) {
        if (id && (e as DOMException).name === 'OverconstrainedError') {
          // the remembered device is gone (unplugged phone/continuity cam) — fall back to default
          chosen.current = '';
          s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        } else throw e;
      }
      // enumerate AFTER permission so labels are filled in; never auto-switch here
      const list = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
      setDevices(list);
      const active = s.getVideoTracks()[0]?.getSettings().deviceId ?? '';
      chosen.current = active;
      setDeviceId(active);

      const el = videoRef.current;
      if (!el) { s.getTracks().forEach(t => t.stop()); throw new Error('video element not mounted'); }
      stream.current = s;
      el.srcObject = s;
      el.muted = true;
      el.playsInline = true;
      await el.play().catch(e => { if ((e as DOMException).name !== 'AbortError') throw e; });
      setStatus('on');
    } catch (e) {
      const err = e as DOMException;
      const msg: Record<string, string> = {
        NotAllowedError: 'Camera permission was denied. Click the camera icon in the address bar and allow it.',
        NotFoundError: 'No camera found on this device.',
        NotReadableError: 'The camera is busy in another app (Zoom/Teams/another tab). Close it and retry.',
        OverconstrainedError: 'The selected camera is unavailable. Pick another one.',
      };
      setError(msg[err?.name] ?? `Camera error: ${err?.message ?? String(e)}`);
      setStatus('error');
      stream.current?.getTracks().forEach(t => t.stop());
      stream.current = null;
    } finally {
      starting.current = false;
    }
  }, [stop, videoRef]);

  const switchTo = useCallback(async (id: string) => {
    chosen.current = id;
    setDeviceId(id);
    if (stream.current) await start(id);
  }, [start]);

  // release the camera on unmount only (deps are stable, so this never re-fires)
  useEffect(() => () => { stream.current?.getTracks().forEach(t => t.stop()); }, []);

  return { status, error, devices, deviceId, start, stop, switchTo };
}
