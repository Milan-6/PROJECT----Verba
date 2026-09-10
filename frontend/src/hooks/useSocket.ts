/**
 * useSocket — one WebSocket to the backend with auto-reconnect (backoff) and typed send helpers.
 * The socket URL is relative, so the Vite dev proxy (and any reverse proxy) handles it.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { ClientMsg, Scenario, ServerMsg } from '../types';

export type ConnStatus = 'connecting' | 'open' | 'closed';

export function useSocket(scenario: Scenario, onMessage: (m: ServerMsg) => void) {
  const [status, setStatus] = useState<ConnStatus>('connecting');
  const ws = useRef<WebSocket | null>(null);
  const cb = useRef(onMessage);
  cb.current = onMessage;
  const retry = useRef(0);
  const timer = useRef<number>();
  const framesDropped = useRef(0);

  useEffect(() => {
    let closed = false;
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const host = import.meta.env.VITE_BACKEND ?? location.host;
      const sock = new WebSocket(`${proto}://${host}/ws/session?scenario=${scenario}`);
      ws.current = sock;
      setStatus('connecting');
      sock.onopen = () => { retry.current = 0; setStatus('open'); };
      sock.onmessage = e => { try { cb.current(JSON.parse(e.data) as ServerMsg); } catch { /* ignore */ } };
      sock.onclose = () => {
        setStatus('closed');
        if (closed) return;
        const delay = Math.min(8000, 500 * 2 ** retry.current++);
        timer.current = window.setTimeout(connect, delay);
      };
      sock.onerror = () => sock.close();
    };
    connect();
    return () => { closed = true; window.clearTimeout(timer.current); ws.current?.close(); };
  }, [scenario]);

  const send = useCallback((m: ClientMsg) => {
    const s = ws.current;
    if (!s || s.readyState !== WebSocket.OPEN) return false;
    // backpressure: if the browser has >64 KB unsent, drop landmark frames rather than queue
    if (m.type === 'frame' && s.bufferedAmount > 65536) { framesDropped.current++; return false; }
    s.send(JSON.stringify(m));
    return true;
  }, []);

  const sendFrame = useCallback((v: Float32Array) => send({ type: 'frame', t: Date.now(), v: Array.from(v, x => Math.round(x * 10000) / 10000) }), [send]);

  return { status, send, sendFrame };
}
