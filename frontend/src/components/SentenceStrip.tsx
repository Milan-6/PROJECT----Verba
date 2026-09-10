import { motion, AnimatePresence } from 'motion/react';
import { Volume2, Trash2, X, Check, Edit3 } from 'lucide-react';
import { STR } from '../i18n';
import type { Lang } from '../types';

interface Props {
  chips: string[];
  lang: Lang;
  pending: string | null;
  spoken: string | null;
  busy: boolean;
  onRemove: (i: number) => void;
  onSpeak: () => void;
  onConfirm: () => void;
  onEdit: () => void;
  onClear: () => void;
}

export default function SentenceStrip({
  chips,
  lang,
  pending,
  spoken,
  busy,
  onRemove,
  onSpeak,
  onConfirm,
  onEdit,
  onClear,
}: Props) {
  const t = STR[lang];

  // The primary sentence to showcase in display serif
  const heroText = pending || spoken || (chips.length > 0 ? chips.map(c => c.replace(/_/g, ' ')).join(' ') : null);

  return (
    <div className="sentence-hero-section">
      <div className="sentence-section-label">
        <span>{t.sentence}</span>
        {chips.length > 0 && (
          <button
            type="button"
            className="btn-outline"
            style={{ padding: '2px 8px', minHeight: '26px', fontSize: '11px', gap: '4px' }}
            onClick={onClear}
          >
            <Trash2 size={12} /> {t.clear}
          </button>
        )}
      </div>

      {/* Level 1 Hero Sentence Reveal */}
      <div style={{ minHeight: '52px', display: 'flex', alignItems: 'center' }}>
        <AnimatePresence mode="wait">
          {heroText ? (
            <motion.p
              key={heroText}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.26, ease: [0.16, 1, 0.3, 1] }}
              className="recognized-sentence-display"
            >
              {heroText}
            </motion.p>
          ) : (
            <motion.p
              key="placeholder"
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.4 }}
              exit={{ opacity: 0 }}
              className="recognized-sentence-display"
              style={{ color: 'var(--ink-faint)', fontStyle: 'italic' }}
            >
              Recognized sentence will appear here…
            </motion.p>
          )}
        </AnimatePresence>
      </div>

      {/* Gloss sequence tokens */}
      <div className="chips-tray">
        {chips.length === 0 ? (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--ink-faint)' }}>
            Sign into camera to add tokens
          </span>
        ) : (
          chips.map((c, i) => (
            <span className="gloss-token-chip" key={i}>
              <span>{c.replace(/_/g, ' ')}</span>
              <button
                type="button"
                className="chip-remove-btn"
                aria-label={`Remove ${c}`}
                onClick={() => onRemove(i)}
              >
                <X size={13} />
              </button>
            </span>
          ))
        )}
      </div>

      {/* Action / Confirm State */}
      {pending ? (
        <div
          className="confirm-box"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--s-2)',
            padding: 'var(--s-3)',
            background: 'var(--canvas-raised)',
            borderRadius: 'var(--r-panel)',
            border: '1px solid var(--amber-glow)',
          }}
        >
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--ink-muted)',
            }}
          >
            {t.confirmSpeak}
          </div>
          <div style={{ display: 'flex', gap: 'var(--s-2)', marginTop: 'var(--s-1)' }}>
            <button className="btn-ink" style={{ flex: 1, minHeight: '44px' }} onClick={onConfirm} autoFocus>
              <Check size={18} /> {t.yes}
            </button>
            <button className="btn-outline" style={{ flex: 1, minHeight: '44px' }} onClick={onEdit}>
              <Edit3 size={16} /> {t.edit}
            </button>
          </div>
        </div>
      ) : (
        <button
          className="btn-ink"
          style={{ width: '100%', minHeight: '44px' }}
          disabled={chips.length === 0 || busy}
          onClick={onSpeak}
        >
          <Volume2 size={20} /> {busy ? 'Synthesizing…' : t.speak}
        </button>
      )}

      {spoken && !pending && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', display: 'flex', gap: 'var(--s-2)' }}>
          <span style={{ textTransform: 'uppercase' }}>{t.spoken}:</span>
          <span>{spoken}</span>
        </div>
      )}
    </div>
  );
}
