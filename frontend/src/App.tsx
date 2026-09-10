import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, Eraser } from 'lucide-react';
import TopBar from './components/TopBar';
import CameraPanel from './components/CameraPanel';
import CandidateCard from './components/CandidateCard';
import SentenceStrip from './components/SentenceStrip';
import HearingInput from './components/HearingInput';
import SignPlayer from './components/SignPlayer';
import { useSocket } from './hooks/useSocket';
import { useSpeech } from './hooks/useSpeech';
import { STR } from './i18n';
import type { Lang, Scenario, ServerMsg, SignItem } from './types';
import type { Quality } from './hooks/useLandmarks';

const saved = <T,>(k: string, d: T): T => { try { return (localStorage.getItem(k) as T) ?? d; } catch { return d; } };

export default function App() {
  const [scenario, setScenario] = useState<Scenario>(saved('sb.scenario', 'hospital'));
  const [lang, setLang] = useState<Lang>(saved('sb.lang', 'en'));
  const t = STR[lang];
  useEffect(() => { try { localStorage.setItem('sb.scenario', scenario); localStorage.setItem('sb.lang', lang); } catch { /* private mode */ } }, [scenario, lang]);
  useEffect(() => { document.documentElement.dataset.scenario = scenario; }, [scenario]);

  // server state
  const [modelReady, setModelReady] = useState(false);
  const [vocabSize, setVocabSize] = useState(0);
  const [presets, setPresets] = useState<string[]>([]);

  // Deaf -> hearing
  const [cameraOn, setCameraOn] = useState(false);
  const [quality, setQuality] = useState<Quality>('no-hands');
  const [candidate, setCandidate] = useState<{ gloss: string; confidence: number } | null>(null);
  const [chips, setChips] = useState<string[]>([]);
  const [pending, setPending] = useState<string | null>(null);
  const [spoken, setSpoken] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);

  // hearing -> Deaf
  const [shown, setShown] = useState<{ text: string; items: SignItem[] }>({ text: '', items: [] });
  const [pinFocused, setPinFocused] = useState(false);
  const [pin, setPin] = useState('');

  const { speak } = useSpeech();
  const chipsRef = useRef(chips); chipsRef.current = chips;

  const onMessage = useCallback((m: ServerMsg) => {
    switch (m.type) {
      case 'hello': setModelReady(m.model); setVocabSize(m.vocab.filter(g => g !== 'NONE' && g !== 'THUMBS_UP').length); setPresets(m.presets); break;
      case 'candidate':
        if (m.gloss === 'THUMBS_UP') { setCandidate(c => { if (c) setChips(x => [...x, c.gloss]); return null; }); break; }
        if (m.gloss === 'DONE') { setCandidate(null); if (chipsRef.current.length) requestSentence(chipsRef.current); break; }
        setCandidate({ gloss: m.gloss, confidence: m.confidence }); break;
      case 'idle': break;
      case 'sign_sequence': setShown({ text: m.text, items: m.items }); break;
      case 'sentence': setBuilding(false); setPending(m.text); break;
      case 'error': console.warn('server:', m.message); break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const { status, send, sendFrame } = useSocket(scenario, onMessage);

  const requestSentence = (g: string[]) => { setBuilding(true); if (!send({ type: 'build_sentence', glosses: g, lang })) setBuilding(false); };

  const confirmSpeak = () => { if (!pending) return; speak(pending, lang); setSpoken(pending); setPending(null); setChips([]); };
  const emergency = () => { const msg = t.emergencySpoken; speak(msg, lang); setSpoken(msg); setPending(null); setChips([]); };
  const clearSession = () => { setChips([]); setPending(null); setSpoken(null); setCandidate(null); setShown({ text: '', items: [] }); setPin(''); send({ type: 'reset' }); window.speechSynthesis?.cancel(); };
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
          <div className="panel-title"><h2>{t.deafPanel}</h2>
            <button className={`btn ${cameraOn ? 'ghost' : 'primary'}`} onClick={() => setCameraOn(o => !o)}>
              {cameraOn ? <CameraOff size={18} /> : <Camera size={18} />} {cameraOn ? t.cameraOff : t.cameraOn}</button></div>
          <CameraPanel active={cameraOn} paused={pinFocused} onVector={onVector} onQuality={setQuality} />
          <div className="quality-line">{cameraOn ? t.quality[quality] : ''}</div>
          <CandidateCard candidate={candidate} lang={lang} modelReady={modelReady}
            onAdd={() => { if (candidate) setChips(c => [...c, candidate.gloss]); setCandidate(null); }}
            onDiscard={() => setCandidate(null)} />
          <SentenceStrip chips={chips} lang={lang} pending={pending} spoken={spoken} busy={building}
            onRemove={i => setChips(c => c.filter((_, k) => k !== i))} onSpeak={() => requestSentence(chips)}
            onConfirm={confirmSpeak} onEdit={() => setPending(null)} onClear={() => { setChips([]); setPending(null); }} />
        </section>

        {/* ---------------- hearing -> Deaf ---------------- */}
        <section className="panel">
          <div className="panel-title"><h2>{t.hearingPanel}</h2>
            <button className="btn ghost" onClick={clearSession}><Eraser size={16} /> {t.clearSession}</button></div>
          <HearingInput lang={lang} presets={presets} disabled={status !== 'open'} onSend={text => send({ type: 'text_in', text, lang })} />
          <SignPlayer text={shown.text} items={shown.items} lang={lang} />
          {scenario === 'bank' && (
            <label className="pin">
              <span>{t.pin}</span>
              <input type="password" inputMode="numeric" maxLength={6} value={pin} onChange={e => setPin(e.target.value.replace(/\D/g, ''))}
                onFocus={() => setPinFocused(true)} onBlur={() => setPinFocused(false)} autoComplete="off" />
            </label>
          )}
        </section>
      </main>

      <footer className="foot"><span>{t.vocabNote(vocabSize)}</span><span>{t.privacy}</span></footer>
    </div>
  );
}
