/**
 * useSocket — one WebSocket to the backend with auto-reconnect (backoff) and typed send helpers.
 * The socket URL is relative, so the Vite dev proxy (and any reverse proxy) handles it.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { ClientMsg, Scenario, ServerMsg } from '../types';
import { getWsBaseUrl, getBackendHost } from '../lib/config';

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
      const wsBase = getWsBaseUrl();
      const wsUrl = `${wsBase}/ws/session?scenario=${scenario}`;
      console.log(`[VERBA-DEV] WebSocket connecting to: ${wsUrl} (host: ${getBackendHost()})`);

      try {
        const sock = new WebSocket(wsUrl);
        ws.current = sock;
        setStatus('connecting');
        sock.onopen = () => {
          retry.current = 0;
          setStatus('open');
          console.log(`[VERBA-DEV] WebSocket connected successfully: ${wsUrl}`);
        };
        sock.onmessage = e => { try { cb.current(JSON.parse(e.data) as ServerMsg); } catch { /* ignore */ } };
        sock.onclose = (ev) => {
          setStatus('closed');
          console.warn(`[VERBA-DEV] WebSocket closed (code: ${ev.code}, reason: ${ev.reason || 'none'}). Reconnecting...`);
          if (closed) return;
          const delay = Math.min(8000, 500 * 2 ** retry.current++);
          timer.current = window.setTimeout(connect, delay);
        };
        sock.onerror = (ev) => {
          console.error(`[VERBA-DEV] WebSocket connection error on: ${wsUrl}`, ev);
          sock.close();
        };
      } catch (err) {
        setStatus('closed');
        console.error(`[VERBA-DEV] WebSocket throw error on: ${wsUrl}`, err);
        if (!closed) {
          const delay = Math.min(8000, 500 * 2 ** retry.current++);
          timer.current = window.setTimeout(connect, delay);
        }
      }
    };
    connect();

    const handleServerChange = () => {
      if (ws.current) ws.current.close();
      retry.current = 0;
      connect();
    };
    window.addEventListener('verba-server-changed', handleServerChange);

    return () => {
      closed = true;
      window.clearTimeout(timer.current);
      window.removeEventListener('verba-server-changed', handleServerChange);
      ws.current?.close();
    };
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
