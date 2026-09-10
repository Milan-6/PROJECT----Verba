/**
 * SignPlayer — what the Deaf person sees when the hearing person sends a message.
 * Priority: 1) the sentence in big high-contrast text  2) the ISL clip sequence, one gloss at a time,
 * with the gloss caption  3) fingerspelling tiles for glosses without a recorded clip.
 */
import { useEffect, useRef, useState } from 'react';
import { RotateCcw, Snail } from 'lucide-react';
import { STR } from '../i18n';
import type { Lang, SignItem } from '../types';

interface Props { text: string; items: SignItem[]; lang: Lang }

export default function SignPlayer({ text, items, lang }: Props) {
  const t = STR[lang];
  const [i, setI] = useState(0);
  const [slow, setSlow] = useState(false);
  const [run, setRun] = useState(0);
  const timer = useRef<number>();
  const vid = useRef<HTMLVideoElement>(null);

  // restart when a new message arrives
  useEffect(() => { setI(0); setRun(r => r + 1); }, [text, items]);

  // advance through items: video -> on ended; letters/question -> timed
  useEffect(() => {
    window.clearTimeout(timer.current);
    const cur = items[i];
    if (!cur) return;
    if (cur.clip) {
      const v = vid.current;
      if (v) { v.playbackRate = slow ? 0.5 : 1; v.currentTime = 0; v.play().catch(() => {}); }
      return;
    }
    const ms = cur.letters ? (slow ? 900 : 550) * Math.max(1, cur.letters.length) : 800;
    timer.current = window.setTimeout(() => setI(n => Math.min(n + 1, items.length)), Math.min(ms, 5000));
    return () => window.clearTimeout(timer.current);
  }, [i, items, slow, run]);

  const cur = items[i];
  const done = i >= items.length;

  return (
    <div className="player">
      <div className="player-head"><span className="muted">{t.shownToDeaf}</span>
        <span className="row">
          <button className="link" onClick={() => setSlow(s => !s)} aria-pressed={slow}><Snail size={14} /> {t.slow}</button>
          <button className="link" onClick={() => { setI(0); setRun(r => r + 1); }}><RotateCcw size={14} /> {t.replay}</button>
        </span></div>
      <div className="bigtext" aria-live="polite">{text || '—'}</div>
      {items.length > 0 && (
        <div className="gloss-line"><span className="muted">{t.glossLabel}:</span> {items.map((it, k) => (
          <span key={k} className={`gloss ${k === i ? 'now' : k < i ? 'done' : ''}`}>{it.gloss.replace(/_/g, ' ')}</span>))}</div>
      )}
      <div className="stage">
        {done || !cur ? (items.length > 0 && <div className="stage-done">✓</div>) : cur.clip ? (
          <video ref={vid} key={`${run}-${i}`} src={cur.clip} muted playsInline onEnded={() => setI(n => n + 1)} className="clip" />
        ) : cur.gloss === '?' ? (
          <div className="fs-tiles"><span className="tile q">?</span></div>
        ) : (
          <div className="fs-wrap">
            <div className="fs-tiles">{cur.letters?.map((c, k) => <span key={k} className={`tile ${c === ' ' ? 'space' : ''}`}>{c.toUpperCase()}</span>)}</div>
            <div className="muted small">{cur.gloss.replace(/_/g, ' ')} · {t.noClip}</div>
          </div>
        )}
      </div>
    </div>
  );
}
