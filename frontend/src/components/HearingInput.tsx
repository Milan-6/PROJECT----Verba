import { useState } from 'react';
import { Mic, MicOff, Send } from 'lucide-react';
import { STR } from '../i18n';
import { useSpeech } from '../hooks/useSpeech';
import type { Lang } from '../types';

interface Props { lang: Lang; presets: string[]; disabled: boolean; onSend: (text: string) => void }

export default function HearingInput({ lang, presets, disabled, onSend }: Props) {
  const t = STR[lang];
  const [text, setText] = useState('');
  const [err, setErr] = useState('');
  const { listen, stopListening, listening, supported } = useSpeech();

  const submit = (s: string) => { const v = s.trim(); if (!v) return; onSend(v); setText(''); };

  const mic = () => {
    setErr('');
    if (listening) { stopListening(); return; }
    listen(lang, (txt, final) => { setText(txt); if (final) submit(txt); }, m => setErr(m));
  };

  return (
    <div className="hearing">
      <form className="row" onSubmit={e => { e.preventDefault(); submit(text); }}>
        <input className="input" value={text} onChange={e => setText(e.target.value)} placeholder={t.typeHere} autoFocus disabled={disabled} aria-label={t.typeHere} />
        <button type="button" className={`btn ${listening ? 'bad' : 'ghost'}`} onClick={mic} disabled={disabled || !supported} title={supported ? '' : 'Not supported in this browser'}>
          {listening ? <MicOff size={20} /> : <Mic size={20} />} {listening ? t.micStop : t.mic}
        </button>
        <button type="submit" className="btn primary" disabled={disabled || !text.trim()}><Send size={20} /> {t.send}</button>
      </form>
      {err && <div className="error-line">{err}</div>}
      <div className="presets">
        <span className="muted">{t.presets}</span>
        {presets.map(p => <button key={p} className="pill" disabled={disabled} onClick={() => submit(p)}>{p}</button>)}
      </div>
    </div>
  );
}
