import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, Eraser } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import TopBar from './components/TopBar';
import CameraPanel from './components/CameraPanel';
import CandidateCard from './components/CandidateCard';
import SentenceStrip from './components/SentenceStrip';
import HearingInput from './components/HearingInput';
import SignPlayer from './components/SignPlayer';
import { useSocket } from './hooks/useSocket';
import { useSpeech } from './hooks/useSpeech';
import { STR } from './i18n';
import type { Lang, Scenario, ServerMsg, SignItem, TranslationStatus } from './types';
import type { Quality, ModelStatus } from './hooks/useLandmarks';

export default function App() {
  const saved = <T,>(k: string, d: T): T => { try { return (localStorage.getItem(k) as unknown as T) || d; } catch { return d; } };
  const [scenario, setScenario] = useState<Scenario>(saved('sb.scenario', 'hospital'));
  const [lang, setLang] = useState<Lang>(saved('sb.lang', 'en'));
  const t = STR[lang];
  useEffect(() => { try { localStorage.setItem('sb.scenario', scenario); localStorage.setItem('sb.lang', lang); } catch { /* private mode */ } }, [scenario, lang]);
  useEffect(() => { document.documentElement.dataset.scenario = scenario; }, [scenario]);

  // server state
  const [modelReady, setModelReady] = useState(false);
  const [visionModelStatus, setVisionModelStatus] = useState<ModelStatus>('loading');
  const [vocabSize, setVocabSize] = useState(0);
  const [presets, setPresets] = useState<string[]>([]);
  const [quickPhrases, setQuickPhrases] = useState<string[]>([]);

  // Deaf -> hearing
  const [cameraOn, setCameraOn] = useState(false);
  const [quality, setQuality] = useState<Quality>('no-hands');
  const [candidate, setCandidate] = useState<{ gloss: string; confidence: number } | null>(null);
  const [chips, setChips] = useState<string[]>([]);
  const [pending, setPending] = useState<string | null>(null);
  const [spoken, setSpoken] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);

  // hearing -> Deaf
  const [shown, setShown] = useState<{
    text: string;
    items: SignItem[];
    translationStatus?: TranslationStatus;
    availableSigns?: string[];
    missingConcepts?: string[];
    fallbackText?: string;
  }>({ text: '', items: [] });
  const [pinFocused, setPinFocused] = useState(false);
  const [pin, setPin] = useState('');

  // crossover cue
  const [crossoverCue, setCrossoverCue] = useState<string | null>(null);

  const triggerCrossover = (msg: string) => {
    setCrossoverCue(msg);
    window.setTimeout(() => setCrossoverCue(null), 1200);
  };

  const { speak } = useSpeech();
  const chipsRef = useRef(chips); chipsRef.current = chips;

  const onMessage = useCallback((m: ServerMsg) => {
    switch (m.type) {
      case 'hello':
        setModelReady(m.model);
        setVocabSize(m.vocab.filter(g => g !== 'NONE' && g !== 'THUMBS_UP').length);
        setPresets(m.presets);
        if ('quick_phrases' in m && Array.isArray(m.quick_phrases)) setQuickPhrases(m.quick_phrases);
        break;
      case 'candidate':
        if (m.gloss === 'THUMBS_UP') { setCandidate(c => { if (c) setChips(x => [...x, c.gloss]); return null; }); break; }
        if (m.gloss === 'DONE') { setCandidate(null); if (chipsRef.current.length) requestSentence(chipsRef.current); break; }
        setCandidate({ gloss: m.gloss, confidence: m.confidence }); break;
      case 'idle': break;
      case 'sign_sequence':
        setShown({
          text: m.text,
          items: m.items,
          translationStatus: m.translation_status,
          availableSigns: m.available_signs,
          missingConcepts: m.missing_concepts,
          fallbackText: m.fallback_text,
        });
        triggerCrossover('ISL Video Ready');
        break;
      case 'sentence': setBuilding(false); setPending(m.text); break;
      case 'error': console.warn('server:', m.message); break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const { status, send, sendFrame } = useSocket(scenario, onMessage);

  const requestSentence = (g: string[]) => { setBuilding(true); if (!send({ type: 'build_sentence', glosses: g, lang })) setBuilding(false); };

  const confirmSpeak = () => {
    if (!pending) return;
    speak(pending, lang);
    setSpoken(pending);
    setPending(null);
    setChips([]);
    triggerCrossover('Voice Synthesized');
  };

  const emergency = () => {
    const msg = t.emergencySpoken;
    speak(msg, lang);
    setSpoken(msg);
    setPending(null);
    setChips([]);
    triggerCrossover('Emergency Alert');
  };

  const clearSession = () => {
    setChips([]);
    setPending(null);
    setSpoken(null);
    setCandidate(null);
    setShown({ text: '', items: [] });
    setPin('');
    send({ type: 'reset' });
    window.speechSynthesis?.cancel();
  };

  const changeScenario = (s: Scenario) => { clearSession(); setScenario(s); };

  const onVector = useCallback((v: Float32Array) => { sendFrame(v); }, [sendFrame]);

  return (
    <div className="app">
      <TopBar scenario={scenario} lang={lang} conn={status} onScenario={changeScenario} onLang={setLang} onEmergency={emergency} />
      {!modelReady && status === 'open' && <div className="banner warn">{t.noModel}</div>}
      {status !== 'open' && <div className="banner">{status === 'connecting' ? t.connecting : t.offline}</div>}

      <main className="workspace">
        {/* ---------------- Deaf -> hearing ---------------- */}
        <section className="panel">
          <div className="panel-head">
            <span className="panel-title">{t.deafPanel}</span>
            <button
              type="button"
              className={cameraOn ? 'btn-outline' : 'btn-ink'}
              style={{ minHeight: '34px', padding: '4px 12px', fontSize: '12px' }}
              onClick={() => setCameraOn(o => !o)}
            >
              {cameraOn ? <CameraOff size={15} /> : <Camera size={15} />} {cameraOn ? t.cameraOff : t.cameraOn}
            </button>
          </div>
          <CameraPanel
            active={cameraOn}
            paused={pinFocused}
            onVector={onVector}
            onQuality={setQuality}
            onModelStatusChange={setVisionModelStatus}
          />
          {cameraOn && quality !== 'good' && (
            <div style={{ minHeight: '1.2em', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-muted)' }}>
              {t.quality[quality]}
            </div>
          )}
          <CandidateCard
            candidate={candidate}
            lang={lang}
            modelReady={modelReady}
            visionModelStatus={visionModelStatus}
            onAdd={() => { if (candidate) setChips(c => [...c, candidate.gloss]); setCandidate(null); }}
            onDiscard={() => setCandidate(null)}
          />
          <SentenceStrip
            chips={chips}
            lang={lang}
            pending={pending}
            spoken={spoken}
            busy={building}
            onRemove={i => setChips(c => c.filter((_, k) => k !== i))}
            onSpeak={() => requestSentence(chips)}
            onConfirm={confirmSpeak}
            onEdit={() => setPending(null)}
            onClear={() => { setChips([]); setPending(null); }}
          />
        </section>

        {/* ---------------- hearing -> Deaf ---------------- */}
        <section className="panel">
          <div className="panel-head">
            <span className="panel-title">{t.hearingPanel}</span>
            <button
              type="button"
              className="btn-outline"
              style={{ minHeight: '34px', padding: '4px 12px', fontSize: '12px' }}
              onClick={clearSession}
            >
              <Eraser size={14} /> {t.clearSession}
            </button>
          </div>
          <HearingInput
            lang={lang}
            presets={presets}
            quickPhrases={quickPhrases}
            disabled={status !== 'open'}
            onSend={text => {
              triggerCrossover('Generating ISL…');
              send({ type: 'text_in', text, lang });
            }}
          />
          <SignPlayer
            text={shown.text}
            items={shown.items}
            lang={lang}
            translationStatus={shown.translationStatus}
            availableSigns={shown.availableSigns}
            missingConcepts={shown.missingConcepts}
            fallbackText={shown.fallbackText}
          />
          {scenario === 'bank' && (
            <label className="pin">
              <span>{t.pin}</span>
              <input
                type="password"
                inputMode="numeric"
                maxLength={6}
                value={pin}
                onChange={e => setPin(e.target.value.replace(/\D/g, ''))}
                onFocus={() => setPinFocused(true)}
                onBlur={() => setPinFocused(false)}
                autoComplete="off"
              />
            </label>
          )}
        </section>
      </main>

      {/* Crossover Cue Overlay */}
      <AnimatePresence>
        {crossoverCue && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: -10 }}
            transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
            style={{
              position: 'fixed',
              bottom: '56px',
              left: '50%',
              transform: 'translateX(-50%)',
              padding: '8px 18px',
              borderRadius: 'var(--r-pill)',
              background: 'var(--ink)',
              color: 'var(--canvas-raised)',
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
              pointerEvents: 'none',
              zIndex: 100,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--amber)' }} />
            {crossoverCue}
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="foot">
        <span>{t.vocabNote(vocabSize)}</span>
        <span>{t.privacy}</span>
      </footer>
    </div>
  );
}
