import { motion, AnimatePresence } from 'motion/react';
import { Check, X } from 'lucide-react';
import { STR } from '../i18n';
import type { Lang } from '../types';
import type { ModelStatus } from '../hooks/useLandmarks';

interface Props {
  candidate: { gloss: string; confidence: number } | null;
  lang: Lang;
  modelReady: boolean;
  visionModelStatus?: ModelStatus;
  onAdd: () => void;
  onDiscard: () => void;
}

const pretty = (g: string) => g.replace(/_/g, ' ');

export default function CandidateCard({ candidate, lang, modelReady, visionModelStatus, onAdd, onDiscard }: Props) {
  const t = STR[lang];
  const isVisionLoading = visionModelStatus === 'loading';
  const isVisionError = visionModelStatus === 'error';

  const statusLabel = () => {
    if (isVisionLoading) return 'Loading Model…';
    if (isVisionError) return 'Model Load Failed';
    if (modelReady) return t.listening;
    return 'Model Offline';
  };

  const dotColor = () => {
    if (isVisionLoading) return 'var(--amber)';
    if (isVisionError) return 'var(--emergency, #dc2626)';
    if (modelReady) return 'var(--ok, #2e7d32)';
    return 'var(--ink-faint)';
  };

  return (
    <div className="candidate-container" style={{ minHeight: '60px' }}>
      <AnimatePresence mode="wait">
        {!candidate ? (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="listening-bar"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--s-3)',
              padding: 'var(--s-3) var(--s-4)',
              background: 'var(--canvas)',
              border: '1px solid var(--hairline)',
              borderRadius: 'var(--r-panel)',
            }}
          >
            <motion.span
              animate={modelReady ? { scale: [1, 1.6, 1], opacity: [0.9, 0.35, 0.9] } : isVisionLoading ? { scale: [1, 1.3, 1], opacity: [0.4, 0.9, 0.4] } : { opacity: 0.4 }}
              transition={modelReady ? { duration: 2, repeat: Infinity, ease: 'easeInOut' } : isVisionLoading ? { duration: 1.2, repeat: Infinity, ease: 'easeInOut' } : undefined}
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: dotColor(),
                display: 'inline-block',
              }}
            />
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--ink-muted)',
              }}
            >
              {statusLabel()}
            </span>
          </motion.div>

        ) : (
          <motion.div
            key="live"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2 }}
            className="candidate-box"
            role="alert"
          >
            <div className="candidate-header-line">
              <span>{t.candidate}</span>
              <span>{Math.round(candidate.confidence * 100)}%</span>
            </div>
            <div className="candidate-gloss-name">{pretty(candidate.gloss)}</div>
            <div className="candidate-confidence-meter">
              <div className="confidence-bar-bg">
                <motion.div
                  className="confidence-bar-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.round(candidate.confidence * 100)}%` }}
                  transition={{ duration: 0.2 }}
                />
              </div>
            </div>
            <div className="candidate-actions-row">
              <button
                className="btn-ink"
                style={{ flex: 1, minHeight: '44px' }}
                onClick={onAdd}
                autoFocus
              >
                <Check size={18} /> {t.add}
              </button>
              <button
                className="btn-outline"
                style={{ flex: 1, minHeight: '44px' }}
                onClick={onDiscard}
              >
                <X size={18} /> {t.discard}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
