import { Check, X } from 'lucide-react';
import { STR } from '../i18n';
import type { Lang } from '../types';

interface Props { candidate: { gloss: string; confidence: number } | null; lang: Lang; modelReady: boolean; onAdd: () => void; onDiscard: () => void }

const pretty = (g: string) => g.replace(/_/g, ' ');

export default function CandidateCard({ candidate, lang, modelReady, onAdd, onDiscard }: Props) {
  const t = STR[lang];
  if (!candidate) {
    return <div className="candidate idle"><span className="pulse" />{modelReady ? t.listening : '—'}</div>;
  }
  const pct = Math.round(candidate.confidence * 100);
  return (
    <div className="candidate live" role="alert">
      <div className="candidate-q">{t.candidate}</div>
      <div className="candidate-gloss">{pretty(candidate.gloss)}</div>
      <div className="conf"><div className="conf-bar" style={{ width: `${pct}%` }} /><span>{pct}%</span></div>
      <div className="candidate-actions">
        <button className="btn ok" onClick={onAdd} autoFocus><Check size={22} /> {t.add}</button>
        <button className="btn bad" onClick={onDiscard}><X size={22} /> {t.discard}</button>
      </div>
    </div>
  );
}
