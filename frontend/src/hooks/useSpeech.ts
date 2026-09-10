/**
 * useSpeech — browser Web Speech API wrappers. Both need no server and upload no audio.
 *   speak(text, lang)     -> speechSynthesis (works offline in Chrome/Edge with installed voices)
 *   listen(lang, onText)  -> SpeechRecognition (Chrome needs internet; typed input is the fallback)
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Lang } from '../types';

const LOCALE: Record<Lang, string> = { en: 'en-IN', hi: 'hi-IN' };

type SR = typeof window extends { SpeechRecognition: infer T } ? T : any;

export function useSpeech() {
  const [speaking, setSpeaking] = useState(false);
  const [listening, setListening] = useState(false);
  const rec = useRef<any>(null);
  const supported = typeof window !== 'undefined' && !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);

  const speak = useCallback((text: string, lang: Lang) => {
    if (!('speechSynthesis' in window) || !text) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = LOCALE[lang];
    u.rate = 0.95;
    const voices = window.speechSynthesis.getVoices();
    const v = voices.find(v => v.lang === LOCALE[lang]) ?? voices.find(v => v.lang.startsWith(lang));
    if (v) u.voice = v;
    u.onstart = () => setSpeaking(true);
    u.onend = u.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(u);
  }, []);

  const stopListening = useCallback(() => { rec.current?.stop(); rec.current = null; setListening(false); }, []);

  const listen = useCallback((lang: Lang, onText: (t: string, final: boolean) => void, onError?: (msg: string) => void) => {
    const Ctor: SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!Ctor) { onError?.('Speech recognition is not supported in this browser. Please type.'); return; }
    stopListening();
    const r = new Ctor();
    r.lang = LOCALE[lang];
    r.interimResults = true;
    r.continuous = false;
    r.onresult = (e: any) => {
      let interim = '', final = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) final += t; else interim += t;
      }
      if (final) onText(final.trim(), true); else if (interim) onText(interim.trim(), false);
    };
    r.onerror = (e: any) => { onError?.(e.error === 'network' ? 'Speech needs internet. Please type instead.' : `Mic error: ${e.error}`); setListening(false); };
    r.onend = () => setListening(false);
    rec.current = r;
    setListening(true);
    r.start();
  }, [stopListening]);

  useEffect(() => () => { rec.current?.stop(); window.speechSynthesis?.cancel(); }, []);

  return { speak, speaking, listen, stopListening, listening, supported };
}
