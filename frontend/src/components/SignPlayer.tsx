/**
 * SignPlayer — High-performance continuous ISL video player with double-buffered preloading.
 *
 * Architecture:
 * - Two video elements (Player A & Player B) eliminate blank screens and loading delays.
 * - While Player A is playing clip N, Player B preloads and buffers clip N+1.
 * - On clip N completion, Player B immediately plays clip N+1 without interruption,
 *   while Player A is assigned clip N+2 to buffer.
 * - Full controls: Play, Pause, Resume, Replay, Previous, Next, Speed toggle (0.75x / 1x).
 * - Clear Missing-Sign UX: Explicitly displays SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED status,
 *   showing available signs vs unavailable concepts with readable fallback text.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  RotateCcw,
  Snail,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  CheckCircle2,
  AlertTriangle,
  XCircle,
} from 'lucide-react';
import { STR } from '../i18n';
import type { Lang, SignItem, TranslationStatus } from '../types';

interface Props {
  text: string;
  items: SignItem[];
  lang: Lang;
  translationStatus?: TranslationStatus;
  availableSigns?: string[];
  missingConcepts?: string[];
  fallbackText?: string;
}

export default function SignPlayer({
  text,
  items,
  lang,
  translationStatus = 'supported',
  availableSigns = [],
  missingConcepts = [],
  fallbackText,
}: Props) {
  const t = STR[lang];

  // Playback state
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isSlow, setIsSlow] = useState(false);
  const [activeBuffer, setActiveBuffer] = useState<'A' | 'B'>('A');
  const [isDelivered, setIsDelivered] = useState(false);

  // Video element references for double buffering
  const videoRefA = useRef<HTMLVideoElement>(null);
  const videoRefB = useRef<HTMLVideoElement>(null);
  const letterTimer = useRef<number>();

  // Reset and trigger edge-glow when a new message/items arrive
  useEffect(() => {
    setIndex(0);
    setIsPlaying(true);
    setActiveBuffer('A');
    if (letterTimer.current) {
      window.clearTimeout(letterTimer.current);
    }
    if (text) {
      setIsDelivered(true);
      const timer = window.setTimeout(() => setIsDelivered(false), 1200);
      return () => window.clearTimeout(timer);
    }
  }, [text, items]);

  const currentItem: SignItem | undefined = items[index];
  const nextItem: SignItem | undefined = items[index + 1];
  const isFinished = index >= items.length && items.length > 0;
  const playbackSpeed = isSlow ? 0.75 : 1.0;

  // Active & Inactive video elements
  const activeVideo = activeBuffer === 'A' ? videoRefA.current : videoRefB.current;
  const inactiveVideo = activeBuffer === 'A' ? videoRefB.current : videoRefA.current;

  // Preload next clip onto the inactive player
  useEffect(() => {
    if (nextItem && nextItem.clip && inactiveVideo) {
      if (inactiveVideo.src !== nextItem.clip) {
        inactiveVideo.src = nextItem.clip;
        inactiveVideo.preload = 'auto';
        inactiveVideo.load();
      }
    }
  }, [nextItem, inactiveVideo]);

  // Setup current clip onto the active player
  useEffect(() => {
    if (!currentItem) return;

    if (currentItem.clip && activeVideo) {
      if (activeVideo.src !== currentItem.clip) {
        activeVideo.src = currentItem.clip;
        activeVideo.load();
      }
      activeVideo.playbackRate = playbackSpeed;
      if (isPlaying) {
        activeVideo.play().catch(() => {
          // Autoplay policy or interrupt
        });
      } else {
        activeVideo.pause();
      }
    } else if (currentItem.letters && isPlaying) {
      // Non-video sign (fingerspelling): time-based advance
      const msPerLetter = isSlow ? 850 : 500;
      const totalMs = Math.min(Math.max(1, currentItem.letters.length) * msPerLetter, 4500);
      window.clearTimeout(letterTimer.current);
      letterTimer.current = window.setTimeout(() => {
        advanceToNext();
      }, totalMs);
    }

    return () => {
      if (letterTimer.current) window.clearTimeout(letterTimer.current);
    };
  }, [index, currentItem, isPlaying, isSlow, activeBuffer]);

  // Advance to next sign with double-buffered transition
  const advanceToNext = useCallback(() => {
    if (index + 1 < items.length) {
      const upcoming = items[index + 1];
      if (upcoming.clip) {
        // Instant switch to pre-buffered player
        setActiveBuffer(prev => (prev === 'A' ? 'B' : 'A'));
      }
      setIndex(prev => prev + 1);
    } else {
      setIndex(items.length);
      setIsPlaying(false);
    }
  }, [index, items]);

  const goToPrevious = () => {
    if (index > 0) {
      setIndex(prev => prev - 1);
      setIsPlaying(true);
      setActiveBuffer(prev => (prev === 'A' ? 'B' : 'A'));
    }
  };

  const restartPlayback = () => {
    setIndex(0);
    setIsPlaying(true);
    setActiveBuffer('A');
  };

  const togglePlayPause = () => {
    if (isFinished) {
      restartPlayback();
      return;
    }
    if (isPlaying) {
      activeVideo?.pause();
      setIsPlaying(false);
      if (letterTimer.current) window.clearTimeout(letterTimer.current);
    } else {
      setIsPlaying(true);
      activeVideo?.play().catch(() => {});
    }
  };

  // Status styling details
  const statusBadge = {
    supported: {
      label: 'Supported in ISL',
      badgeClass: 'badge-ok',
      icon: <CheckCircle2 size={13} />,
    },
    partially_supported: {
      label: 'Partially Supported',
      badgeClass: 'badge-warn',
      icon: <AlertTriangle size={13} />,
    },
    unsupported: {
      label: 'Unavailable',
      badgeClass: 'badge-error',
      icon: <XCircle size={13} />,
    },
  }[translationStatus] || {
    label: 'Translation',
    badgeClass: '',
    icon: null,
  };

  const progressPercent = items.length > 0 ? Math.min(100, Math.round(((index + 1) / items.length) * 100)) : 0;

  return (
    <div className={`sign-display-panel ${isDelivered ? 'delivered' : ''}`}>
      {/* Player Header with Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-2)' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ink-muted)' }}>
            {t.shownToDeaf}
          </span>
          <span className={`status-badge ${statusBadge.badgeClass}`}>
            {statusBadge.icon}
            <span>{statusBadge.label}</span>
          </span>
        </div>
        {items.length > 0 && (
          <button
            type="button"
            className="btn-outline"
            style={{ padding: '2px 8px', minHeight: '26px', fontSize: '11px', gap: '4px' }}
            onClick={restartPlayback}
            title="Restart playback"
          >
            <RotateCcw size={12} /> {t.replay}
          </button>
        )}
      </div>

      {/* Main High-Contrast Text for Deaf Patient */}
      <div
        className="bigtext"
        aria-live="polite"
        style={{
          fontFamily: 'var(--font-serif)',
          fontSize: '22px',
          fontWeight: 600,
          color: 'var(--ink)',
          lineHeight: 1.3,
          minHeight: '30px',
        }}
      >
        {text || '—'}
      </div>

      {/* Missing Concepts & Translation Breakdown */}
      {missingConcepts.length > 0 && (
        <div className="missing-alert">
          <div className="missing-title">
            <AlertTriangle size={15} />
            <span>Some signs are unavailable for this sentence:</span>
          </div>
          <div className="missing-breakdown">
            {availableSigns.length > 0 && (
              <div className="breakdown-group">
                <span className="breakdown-label">Available:</span>
                <span className="breakdown-chips">
                  {availableSigns.map((s, i) => (
                    <span key={i} className="chip-avail">
                      {s.replace(/_/g, ' ')}
                    </span>
                  ))}
                </span>
              </div>
            )}
            <div className="breakdown-group">
              <span className="breakdown-label">Unavailable:</span>
              <span className="breakdown-chips">
                {missingConcepts.map((s, i) => (
                  <span key={i} className="chip-unavail">
                    {s.replace(/_/g, ' ')}
                  </span>
                ))}
              </span>
            </div>
          </div>
          {fallbackText && <div className="fallback-text">{fallbackText}</div>}
        </div>
      )}

      {/* Gloss Sequence Strip with Interactive States */}
      {items.length > 0 && (
        <div className="gloss-sequence-container" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-2)' }}>
          <div className="gloss-timeline">
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase', color: 'var(--ink-muted)', marginRight: 4 }}>
              {t.glossLabel}:
            </span>
            {items.map((it, k) => (
              <button
                key={k}
                type="button"
                className={`gloss-timeline-token ${k === index ? 'now' : k < index ? 'done' : ''}`}
                onClick={() => {
                  setIndex(k);
                  setIsPlaying(true);
                  setActiveBuffer(prev => (prev === 'A' ? 'B' : 'A'));
                }}
                title={`Jump to sign ${k + 1}: ${it.gloss}`}
              >
                {it.gloss.replace(/_/g, ' ')}
              </button>
            ))}
          </div>

          {/* Linear Progress Bar */}
          <div className="progress-bar-wrap">
            <div
              className="progress-bar-fill"
              style={{ width: `${progressPercent}%` }}
              role="progressbar"
              aria-valuenow={progressPercent}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </div>
      )}

      {/* Visual Sign Display Stage (Continuous Double-Buffered Player) */}
      <div className="video-stage-container">
        {isFinished ? (
          <div className="stage-done-card">
            <CheckCircle2 size={40} className="icon-success" />
            <div className="stage-done-text">Sign Sequence Complete</div>
            <button className="btn-ink" onClick={restartPlayback} style={{ marginTop: 'var(--s-2)', minHeight: '38px' }}>
              <RotateCcw size={15} /> Replay
            </button>
          </div>
        ) : currentItem?.clip ? (
          <>
            {/* Double-buffered video players */}
            <video
              ref={videoRefA}
              className={activeBuffer === 'A' ? 'visible' : 'hidden-preload'}
              muted
              playsInline
              onEnded={advanceToNext}
              onError={() => advanceToNext()}
            />
            <video
              ref={videoRefB}
              className={activeBuffer === 'B' ? 'visible' : 'hidden-preload'}
              muted
              playsInline
              onEnded={advanceToNext}
              onError={() => advanceToNext()}
            />
            <div className="clip-caption">
              <span>
                Sign {index + 1} / {items.length}
              </span>
              <span className="sign-name">{currentItem.gloss.replace(/_/g, ' ')}</span>
            </div>
          </>
        ) : currentItem?.gloss === '?' ? (
          <div className="fs-tiles" style={{ height: '100%', alignItems: 'center' }}>
            <span className="tile q">?</span>
          </div>
        ) : currentItem ? (
          <div className="fs-wrap" style={{ height: '100%', justifyContent: 'center' }}>
            <div className="fs-tiles">
              {currentItem.letters?.map((c, k) => (
                <span key={k} className={`tile ${c === ' ' ? 'space' : ''}`}>
                  {c.toUpperCase()}
                </span>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)', marginTop: 'var(--s-2)' }}>
              {currentItem.gloss.replace(/_/g, ' ')} · {t.noClip}
            </div>
          </div>
        ) : (
          <div className="empty-stage-message">
            <span>Waiting for speech or text input…</span>
          </div>
        )}
      </div>

      {/* Playback Control Bar */}
      {items.length > 0 && (
        <div className="player-controls-row">
          <div className="transport-btns">
            <button
              type="button"
              className="transport-ctrl-btn"
              onClick={goToPrevious}
              disabled={index === 0}
              title="Previous sign"
            >
              <SkipBack size={16} />
            </button>
            <button
              type="button"
              className="transport-ctrl-btn primary-ctrl"
              onClick={togglePlayPause}
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button
              type="button"
              className="transport-ctrl-btn"
              onClick={advanceToNext}
              disabled={index >= items.length - 1}
              title="Next sign"
            >
              <SkipForward size={16} />
            </button>
            <button
              type="button"
              className={`speed-toggle-pill ${isSlow ? 'active' : ''}`}
              onClick={() => setIsSlow(s => !s)}
              aria-pressed={isSlow}
              title={isSlow ? 'Play at 1.0x' : 'Play at 0.75x'}
            >
              <Snail size={13} style={{ display: 'inline', marginRight: 4 }} />
              {isSlow ? '0.75x' : '1.0x'}
            </button>
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)' }}>
            {index < items.length ? `Sign ${index + 1} of ${items.length}` : 'Done'}
          </span>
        </div>
      )}

      {/* Linguistic & Playback Limitations Disclosure */}
      <div
        style={{
          marginTop: 'var(--s-2)',
          paddingTop: 'var(--s-2)',
          fontSize: '11px',
          fontFamily: 'var(--font-mono)',
          color: 'var(--ink-faint)',
          borderTop: '1px solid var(--hairline)',
          lineHeight: 1.4,
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
        }}
      >
        <div>
          <strong style={{ color: 'var(--ink-muted)' }}>Linguistic Limitation:</strong> Non-manual markers (eyebrow raises, head tilts, mouth shapes) are not modeled by landmark tracking; grammar cues are represented via UI badges.
        </div>
        <div>
          <strong style={{ color: 'var(--ink-muted)' }}>Playback Notice:</strong> Continuous playback uses double-buffered preloading; seamless clip handoff avoids blank frames.
        </div>
      </div>
    </div>
  );
}
