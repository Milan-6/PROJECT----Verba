import { Building2, HeartPulse, Languages, Siren } from 'lucide-react';
import { STR } from '../i18n';
import type { Lang, Scenario } from '../types';
import type { ConnStatus } from '../hooks/useSocket';

interface Props { scenario: Scenario; lang: Lang; conn: ConnStatus; onScenario: (s: Scenario) => void; onLang: (l: Lang) => void; onEmergency: () => void }

export default function TopBar({ scenario, lang, conn, onScenario, onLang, onEmergency }: Props) {
  const t = STR[lang];
  return (
    <header className="topbar">
      <div className="brand"><span className="logo">SB</span><div><b>{t.appName}</b><small>{t.tagline}</small></div></div>
      <div className="seg" role="tablist">
        <button role="tab" aria-selected={scenario === 'hospital'} className={scenario === 'hospital' ? 'on' : ''} onClick={() => onScenario('hospital')}><HeartPulse size={18} /> {t.hospital}</button>
        <button role="tab" aria-selected={scenario === 'bank'} className={scenario === 'bank' ? 'on' : ''} onClick={() => onScenario('bank')}><Building2 size={18} /> {t.bank}</button>
      </div>
      <div className="right">
        <span className={`dot ${conn}`} title={conn === 'open' ? t.connected : conn === 'connecting' ? t.connecting : t.offline} />
        <button className="btn ghost" onClick={() => onLang(lang === 'en' ? 'hi' : 'en')}><Languages size={18} /> {lang === 'en' ? 'हिन्दी' : 'English'}</button>
        {scenario === 'hospital' && <button className="btn emergency" onClick={onEmergency}><Siren size={20} /> {t.emergency}</button>}
      </div>
    </header>
  );
}
