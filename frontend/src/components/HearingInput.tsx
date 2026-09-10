import { useState } from 'react';
import { Mic, MicOff, Send, Building2, Stethoscope, Sparkles } from 'lucide-react';
import { STR } from '../i18n';
import { useSpeech } from '../hooks/useSpeech';
import type { Lang } from '../types';

interface Props {
  lang: Lang;
  presets: string[];
  disabled: boolean;
  onSend: (text: string) => void;
}

export const BANKING_QUICK_PHRASES = [
  'Please enter your PIN',
  'Sign here on the form',
  'Please show your ID proof',
  'Insert your chip card',
  'How much do you want to withdraw?',
  'Please collect your receipt',
];

export const HOSPITAL_QUICK_PHRASES = [
  'Where does it hurt?',
  'Since how many days?',
  'Please show your insurance card',
  'The doctor will see you now',
  'Please take a seat',
  'Do you have any allergy?',
];

export default function HearingInput({ lang, presets: _presets, disabled, onSend }: Props) {
  const t = STR[lang];
  const [text, setText] = useState('');
  const [err, setErr] = useState('');
  const [activeTab, setActiveTab] = useState<'all' | 'bank' | 'hospital'>('all');
  const { listen, stopListening, listening, supported } = useSpeech();

  const submit = (s: string) => {
    const v = s.trim();
    if (!v) return;
    onSend(v);
    setText('');
  };

  const mic = () => {
    setErr('');
    if (listening) {
      stopListening();
      return;
    }
    listen(
      lang,
      (txt, final) => {
        setText(txt);
        if (final) submit(txt);
      },
      m => setErr(m)
    );
  };

  return (
    <div className="hearing-panel-content" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-3)' }}>
      <form
        className="hearing-input-form"
        onSubmit={e => {
          e.preventDefault();
          submit(text);
        }}
      >
        <input
          className="text-input-field"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder={t.typeHere}
          autoFocus
          disabled={disabled}
          aria-label={t.typeHere}
        />
        <button
          type="button"
          className={listening ? 'btn-emergency' : 'btn-outline'}
          style={{ minHeight: '44px', whiteSpace: 'nowrap' }}
          onClick={mic}
          disabled={disabled || !supported}
          title={supported ? '' : 'Not supported in this browser'}
        >
          {listening ? <MicOff size={18} /> : <Mic size={18} />} {listening ? t.micStop : t.mic}
        </button>
        <button
          type="submit"
          className="btn-ink"
          style={{ minHeight: '44px', whiteSpace: 'nowrap' }}
          disabled={disabled || !text.trim()}
        >
          <Send size={18} /> {t.send}
        </button>
      </form>
      {err && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--emergency)' }}>
          {err}
        </div>
      )}

      {/* Quick Phrases Section */}
      <div className="quick-phrases-wrapper">
        <div className="quick-phrases-toolbar">
          <div className="quick-phrases-label">
            <Sparkles size={13} style={{ color: 'var(--amber)' }} />
            <span>Target Quick Phrases (12)</span>
          </div>
          <div className="quick-category-seg">
            <button
              type="button"
              className={`quick-cat-btn ${activeTab === 'all' ? 'active' : ''}`}
              onClick={() => setActiveTab('all')}
            >
              All (12)
            </button>
            <button
              type="button"
              className={`quick-cat-btn ${activeTab === 'bank' ? 'active' : ''}`}
              onClick={() => setActiveTab('bank')}
            >
              <Building2 size={11} style={{ display: 'inline', marginRight: 4 }} />
              Bank (6)
            </button>
            <button
              type="button"
              className={`quick-cat-btn ${activeTab === 'hospital' ? 'active' : ''}`}
              onClick={() => setActiveTab('hospital')}
            >
              <Stethoscope size={11} style={{ display: 'inline', marginRight: 4 }} />
              Hospital (6)
            </button>
          </div>
        </div>

        {/* Grouped Banking Quick Phrases */}
        {(activeTab === 'all' || activeTab === 'bank') && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-1)' }}>
            {activeTab === 'all' && (
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  fontWeight: 600,
                  color: 'var(--ink-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--s-1)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                }}
              >
                <Building2 size={12} /> Banking / ATM
              </span>
            )}
            <div className="presets" style={{ gap: 'var(--s-2)' }}>
              {BANKING_QUICK_PHRASES.map(p => (
                <button
                  key={p}
                  type="button"
                  className="quick-phrase-pill-btn"
                  disabled={disabled}
                  onClick={() => submit(p)}
                  title="Tap to translate to ISL video playback"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Grouped Hospital Quick Phrases */}
        {(activeTab === 'all' || activeTab === 'hospital') && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--s-1)',
              marginTop: activeTab === 'all' ? 'var(--s-2)' : '0',
            }}
          >
            {activeTab === 'all' && (
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  fontWeight: 600,
                  color: 'var(--ink-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--s-1)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                }}
              >
                <Stethoscope size={12} /> Hospital / Medical
              </span>
            )}
            <div className="presets" style={{ gap: 'var(--s-2)' }}>
              {HOSPITAL_QUICK_PHRASES.map(p => (
                <button
                  key={p}
                  type="button"
                  className="quick-phrase-pill-btn"
                  disabled={disabled}
                  onClick={() => submit(p)}
                  title="Tap to translate to ISL video playback"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
