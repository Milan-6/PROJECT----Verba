import { useState } from 'react';
import { Mic, MicOff, Send, Sparkles, MessageSquareQuote } from 'lucide-react';
import { STR } from '../i18n';
import { useSpeech } from '../hooks/useSpeech';
import type { Lang } from '../types';

interface Props {
  lang: Lang;
  presets?: string[];
  quickPhrases?: string[];
  disabled: boolean;
  onSend: (text: string) => void;
}

export default function HearingInput({ lang, presets = [], quickPhrases = [], disabled, onSend }: Props) {
  const t = STR[lang];
  const [text, setText] = useState('');
  const [err, setErr] = useState('');
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

  const hasQuickPhrases = Array.isArray(quickPhrases) && quickPhrases.length > 0;
  const hasPresets = Array.isArray(presets) && presets.length > 0;

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

      {/* Feature 4: Scenario-Aware High-Frequency Quick Phrases (100% Video-Backed) */}
      {hasQuickPhrases && (
        <div className="quick-phrases-wrapper" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-2)' }}>
          <div className="quick-phrases-label" style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-2)' }}>
            <Sparkles size={13} style={{ color: 'var(--amber)' }} />
            <span>{t.quickPhrases} ({quickPhrases.length})</span>
          </div>
          <div className="presets" style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--s-2)' }}>
            {quickPhrases.map(p => (
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

      {/* Feature 3: Scenario Preset Suggestions (100% Video-Backed) */}
      {hasPresets && (
        <div className="suggested-sentences-wrapper" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-2)' }}>
          <div className="quick-phrases-label" style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-2)' }}>
            <MessageSquareQuote size={13} style={{ color: 'var(--ink-muted)' }} />
            <span>{t.suggestedSentences} ({presets.length})</span>
          </div>
          <div className="presets" style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--s-2)' }}>
            {presets.map(p => (
              <button
                key={p}
                type="button"
                className="quick-phrase-pill-btn preset-phrase-btn"
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
  );
}

