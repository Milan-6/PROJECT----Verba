/**
 * Centralized VERBA Backend Configuration & Diagnostics
 * Resolves the backend host, HTTP base URL, and WebSocket base URL.
 */

export const DEFAULT_DEV_HOST = '10.52.179.93:8000';
const STORAGE_KEY = 'sb.backend';

/** Clean and normalize a host string (removes protocol, trailing slashes, etc.) */
export function normalizeHost(input: string): string {
  let cleaned = input.trim();
  cleaned = cleaned.replace(/^https?:\/\//i, '');
  cleaned = cleaned.replace(/^wss?:\/\//i, '');
  cleaned = cleaned.replace(/\/+$/, '');
  return cleaned;
}

/** Check if currently running inside Capacitor / Mobile WebView */
export function isCapacitorApp(): boolean {
  if (typeof window === 'undefined') return false;
  return Boolean(
    (window as any).Capacitor ||
    (window as any).VerbaNative ||
    (location.hostname === 'localhost' && location.port === '')
  );
}

/**
 * Resolves the backend host (IP:PORT or domain).
 * Priority:
 * 1. Saved in localStorage ('sb.backend')
 * 2. Vite env var (VITE_BACKEND or VITE_API_BASE_URL)
 * 3. Default LAN IP for Capacitor (10.52.179.93:8000)
 * 4. Current browser host (location.host)
 */
export function getBackendHost(): string {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && saved.trim()) {
      return normalizeHost(saved);
    }
  }

  const envHost = import.meta.env.VITE_BACKEND || import.meta.env.VITE_API_BASE_URL;
  if (envHost && String(envHost).trim()) {
    return normalizeHost(String(envHost));
  }

  if (isCapacitorApp()) {
    return DEFAULT_DEV_HOST;
  }

  if (typeof window !== 'undefined' && location.host) {
    return location.host;
  }

  return DEFAULT_DEV_HOST;
}

/** Determines whether a host should use plain (unencrypted) HTTP/WS */
function isPlainTraffic(host: string): boolean {
  return /^(\d+\.\d+\.\d+\.\d+|localhost)(:\d+)?$/.test(host);
}

/** Returns the HTTP base URL e.g. "http://10.52.179.93:8000" */
export function getHttpBaseUrl(): string {
  const host = getBackendHost();
  const proto = isPlainTraffic(host) ? 'http' : (typeof window !== 'undefined' && location.protocol === 'https:' ? 'https' : 'http');
  return `${proto}://${host}`;
}

/** Returns the WebSocket base URL e.g. "ws://10.52.179.93:8000" */
export function getWsBaseUrl(): string {
  const host = getBackendHost();
  const proto = isPlainTraffic(host) ? 'ws' : (typeof window !== 'undefined' && location.protocol === 'https:' ? 'wss' : 'ws');
  return `${proto}://${host}`;
}

/** Save a new backend host to localStorage and notify the app */
export function saveBackendHost(host: string): void {
  const normalized = normalizeHost(host);
  if (normalized) {
    localStorage.setItem(STORAGE_KEY, normalized);
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
  window.dispatchEvent(new Event('verba-server-changed'));
}

/** Reset backend host to default */
export function resetBackendHost(): void {
  localStorage.removeItem(STORAGE_KEY);
  window.dispatchEvent(new Event('verba-server-changed'));
}

export interface DiagnosticResult {
  host: string;
  httpUrl: string;
  wsUrl: string;
  httpOk: boolean;
  httpStatus: number | null;
  modelOnline: boolean | null;
  latencyMs: number | null;
  errorType: string | null;
  errorDetail: string | null;
}

/** Lightweight diagnostic check: tests HTTP /health endpoint */
export async function testBackendConnection(): Promise<DiagnosticResult> {
  const host = getBackendHost();
  const httpUrl = getHttpBaseUrl();
  const wsUrl = getWsBaseUrl();
  const healthUrl = `${httpUrl}/health`;

  const result: DiagnosticResult = {
    host,
    httpUrl,
    wsUrl,
    httpOk: false,
    httpStatus: null,
    modelOnline: null,
    latencyMs: null,
    errorType: null,
    errorDetail: null,
  };

  const start = performance.now();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);

  try {
    const res = await fetch(healthUrl, {
      method: 'GET',
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    clearTimeout(timeoutId);
    result.latencyMs = Math.round(performance.now() - start);
    result.httpStatus = res.status;

    if (res.ok) {
      result.httpOk = true;
      const data = await res.json().catch(() => ({}));
      result.modelOnline = Boolean(data.model);
    } else {
      result.errorType = `HTTP_${res.status}`;
      result.errorDetail = `Server returned status ${res.status}: ${res.statusText}`;
    }
  } catch (err: any) {
    clearTimeout(timeoutId);
    result.latencyMs = Math.round(performance.now() - start);
    if (err.name === 'AbortError') {
      result.errorType = 'TIMEOUT';
      result.errorDetail = 'Request timed out after 4 seconds (check IP/firewall/Wi-Fi)';
    } else if (err.message && err.message.includes('Failed to fetch')) {
      result.errorType = 'CONNECTION_REFUSED_OR_CORS';
      result.errorDetail = 'Network request failed (connection refused, wrong IP, or unreachable)';
    } else {
      result.errorType = err.name || 'UNKNOWN_ERROR';
      result.errorDetail = err.message || String(err);
    }
  }

  return result;
}
