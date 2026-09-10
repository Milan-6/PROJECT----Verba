import { Volume2, Trash2, X } from 'lucide-react';
import { STR } from '../i18n';
import type { Lang } from '../types';

interface Props {
  chips: string[]; lang: Lang; pending: string | null; spoken: string | null; busy: boolean;
  onRemove: (i: number) => void; onSpeak: () => void; onConfirm: () => void; onEdit: () => void; onClear: () => void;
}

export default function SentenceStrip({ chips, lang, pending, spoken, busy, onRemove, onSpeak, onConfirm, onEdit, onClear }: Props) {
  const t = STR[lang];
  return (
    <div className="strip">
      <div className="strip-head"><span>{t.sentence}</span>
        {chips.length > 0 && <button className="link" onClick={onClear}><Trash2 size={14} /> {t.clear}</button>}</div>
      <div className="chips">
        {chips.length === 0 && <span className="muted">…</span>}
        {chips.map((c, i) => (
          <span className="chip" key={i}>{c.replace(/_/g, ' ')}<button aria-label="remove" onClick={() => onRemove(i)}><X size={14} /></button></span>
        ))}
      </div>
      {pending ? (
        <div className="confirm">
          <div className="confirm-q">{t.confirmSpeak}</div>
          <div className="confirm-text">{pending}</div>
          <div className="row">
            <button className="btn ok" onClick={onConfirm} autoFocus><Volume2 size={20} /> {t.yes}</button>
            <button className="btn ghost" onClick={onEdit}>{t.edit}</button>
          </div>
        </div>
      ) : (
        <button className="btn primary big" disabled={chips.length === 0 || busy} onClick={onSpeak}><Volume2 size={22} /> {t.speak}</button>
      )}
      {spoken && !pending && <div className="spoken"><span className="muted">{t.spoken}:</span> {spoken}</div>}
    </div>
  );
}
