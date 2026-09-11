import { motion } from 'motion/react';
import { Building2, HeartPulse, Bus, Train, ShoppingBag, Landmark, Languages, Siren } from 'lucide-react';
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
  const modes: { key: Scenario; icon: any; label: string }[] = [
    { key: 'hospital', icon: HeartPulse, label: t.hospital },
    { key: 'bank', icon: Building2, label: t.bank },
    { key: 'bus_stop', icon: Bus, label: t.bus_stop },
    { key: 'railway', icon: Train, label: t.railway },
    { key: 'shopping', icon: ShoppingBag, label: t.shopping },
    { key: 'government', icon: Landmark, label: t.government },
  ];

  return (
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
        {/* Connection status indicator: quiet amber/neutral, not generic green */}
        <div
          className="status-indicator"
          title={conn === 'open' ? t.connected : conn === 'connecting' ? t.connecting : t.offline}
        >
          <span className={`status-dot ${conn}`} />
          <span className="status-label" style={{ display: 'none' }}>{conn}</span>
        </div>

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
  );
}
