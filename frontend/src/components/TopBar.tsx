import { motion } from 'motion/react';
import { Building2, HeartPulse, Languages, Siren } from 'lucide-react';
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
  const modes: Scenario[] = ['hospital', 'bank'];

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
          const isActive = scenario === m;
          return (
            <button
              key={m}
              role="tab"
              type="button"
              aria-selected={isActive}
              className={`mode-tab-btn ${isActive ? 'active' : ''}`}
              onClick={() => onScenario(m)}
            >
              {m === 'hospital' ? <HeartPulse size={16} /> : <Building2 size={16} />}
              {m === 'hospital' ? t.hospital : t.bank}
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
