import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Building2, HeartPulse, Bus, Train, ShoppingBag, Landmark, Languages, Siren, Server, Wifi, X, Check } from 'lucide-react';
import { STR } from '../i18n';
import type { Lang, Scenario } from '../types';
import type { ConnStatus } from '../hooks/useSocket';

interface Props {
  scenario: Scenario;
  lang: Lang;
  conn: ConnStatus;
  onScenario: (s: Scenario) => void;
  onLang: (l: Lang) => void;
  onEmergency: () => void;
}

export default function TopBar({ scenario, lang, conn, onScenario, onLang, onEmergency }: Props) {
  const t = STR[lang];
  const [showModal, setShowModal] = useState(false);
  const [ipInput, setIpInput] = useState('');

  const openModal = () => {
    const saved = localStorage.getItem('sb.backend');
    if (saved) {
      setIpInput(saved);
    } else {
      const isCapacitor = typeof window !== 'undefined' && ((window as any).Capacitor || (window as any).VerbaNative || location.hostname === 'localhost');
      setIpInput(isCapacitor ? '192.168.241.251:8000' : (import.meta.env.VITE_BACKEND ?? location.host));
    }
    setShowModal(true);
  };

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    if (ipInput.trim()) {
      localStorage.setItem('sb.backend', ipInput.trim());
    } else {
      localStorage.removeItem('sb.backend');
    }
    window.dispatchEvent(new Event('verba-server-changed'));
    setShowModal(false);
  };

  const handleReset = () => {
    localStorage.removeItem('sb.backend');
    setIpInput('192.168.241.251:8000');
    window.dispatchEvent(new Event('verba-server-changed'));
    setShowModal(false);
  };

  const modes: { key: Scenario; icon: any; label: string }[] = [
    { key: 'hospital', icon: HeartPulse, label: t.hospital },
    { key: 'bank', icon: Building2, label: t.bank },
    { key: 'bus_stop', icon: Bus, label: t.bus_stop },
    { key: 'railway', icon: Train, label: t.railway },
    { key: 'shopping', icon: ShoppingBag, label: t.shopping },
    { key: 'government', icon: Landmark, label: t.government },
  ];

  return (
    <>
      <header className="topbar">
        {/* Brand: Display serif wordmark + quiet monospace tagline */}
        <div className="brand">
          <span className="brand-wordmark">{t.appName}</span>
          <span className="brand-tagline">{t.tagline}</span>
        </div>

        {/* Mode Tabs: Gliding underline spring animation (Section 6 reference) */}
        <nav className="mode-tabs-container" role="tablist" aria-label="Scenario modes">
          {modes.map(m => {
            const isActive = scenario === m.key;
            const Icon = m.icon;
            return (
              <button
                key={m.key}
                role="tab"
                type="button"
                aria-selected={isActive}
                className={`mode-tab-btn ${isActive ? 'active' : ''}`}
                onClick={() => onScenario(m.key)}
              >
                <Icon size={16} />
                {m.label}
                {isActive && (
                  <motion.div
                    layoutId="mode-underline"
                    className="mode-tab-underline"
                    transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Level-3 Controls: Connection dot, Language pill, Solid Emergency */}
        <div className="topbar-right">
          {/* Connection status indicator: clickable to configure backend IP */}
          <button
            type="button"
            className="status-indicator"
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: '4px' }}
            onClick={openModal}
            title={`Status: ${conn === 'open' ? t.connected : conn === 'connecting' ? t.connecting : t.offline} — Tap to configure server IP`}
            aria-label="Configure Server"
          >
            <span className={`status-dot ${conn}`} />
            <span className="status-label" style={{ display: 'none' }}>{conn}</span>
          </button>

          {/* Quiet outlined language toggle */}
          <button
            type="button"
            className="lang-pill-btn"
            onClick={() => onLang(lang === 'en' ? 'hi' : 'en')}
            title="Switch language"
          >
            <Languages size={14} />
            <span>{lang === 'en' ? 'हिन्दी' : 'English'}</span>
          </button>

          {/* Emergency button: Solid, high-contrast, preserved exact red */}
          {scenario === 'hospital' && (
            <button type="button" className="btn-emergency" onClick={onEmergency}>
              <Siren size={16} />
              <span>{t.emergency}</span>
            </button>
          )}
        </div>
      </header>

      {/* Server Config Modal */}
      <AnimatePresence>
        {showModal && (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.65)',
              backdropFilter: 'blur(4px)',
              zIndex: 9999,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '16px',
            }}
            onClick={() => setShowModal(false)}
          >
            <motion.div
              initial={{ scale: 0.92, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.92, opacity: 0 }}
              transition={{ type: 'spring', duration: 0.25 }}
              style={{
                backgroundColor: 'var(--bg-card, #1c1c1f)',
                border: '1px solid var(--border, #2e2e34)',
                borderRadius: '16px',
                padding: '24px',
                maxWidth: '420px',
                width: '100%',
                boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
                color: 'var(--ink, #ffffff)',
              }}
              onClick={e => e.stopPropagation()}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Server size={20} style={{ color: 'var(--accent, #6366f1)' }} />
                  <span style={{ fontWeight: 600, fontSize: '16px' }}>Backend Server</span>
                </div>
                <button
                  type="button"
                  style={{ background: 'transparent', border: 'none', color: 'var(--ink-muted, #888)', cursor: 'pointer', padding: '4px' }}
                  onClick={() => setShowModal(false)}
                >
                  <X size={18} />
                </button>
              </div>

              <p style={{ fontSize: '13px', color: 'var(--ink-muted, #a1a1aa)', marginBottom: '16px', lineHeight: 1.4 }}>
                Set the host address of your VERBA backend on your local Wi-Fi network (e.g. your PC's IP and port).
              </p>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', fontSize: '12px', padding: '8px 12px', borderRadius: '8px', background: 'rgba(255,255,255,0.05)' }}>
                <Wifi size={14} style={{ color: conn === 'open' ? 'var(--ok, #10b981)' : 'var(--amber, #f59e0b)' }} />
                <span>Current Status: <strong style={{ color: conn === 'open' ? '#10b981' : '#f59e0b' }}>{conn.toUpperCase()}</strong></span>
              </div>

              <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, marginBottom: '6px', color: 'var(--ink-muted, #a1a1aa)' }}>
                    Host & Port (IP:PORT)
                  </label>
                  <input
                    type="text"
                    value={ipInput}
                    onChange={e => setIpInput(e.target.value)}
                    placeholder="192.168.241.251:8000"
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      borderRadius: '8px',
                      border: '1px solid var(--border, #3f3f46)',
                      background: 'rgba(0,0,0,0.3)',
                      color: '#ffffff',
                      fontSize: '14px',
                      outline: 'none',
                    }}
                  />
                </div>

                <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '4px' }}>
                  <button
                    type="button"
                    onClick={handleReset}
                    className="btn-outline"
                    style={{ padding: '8px 14px', fontSize: '12px' }}
                  >
                    Reset Default
                  </button>
                  <button
                    type="submit"
                    className="btn-ink"
                    style={{ padding: '8px 16px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    <Check size={14} /> Save & Reconnect
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
