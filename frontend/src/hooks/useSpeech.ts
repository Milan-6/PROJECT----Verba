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
  const hasNative = typeof window !== 'undefined' && !!(window as any).VerbaNative;
  const supported = hasNative || (typeof window !== 'undefined' && !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition));

  useEffect(() => {
    if (hasNative) {
      (window as any).onVerbaTTSStatus = (active: boolean) => setSpeaking(active);
    }
  }, [hasNative]);

  const speak = useCallback((text: string, lang: Lang) => {
    if (!text) return;
    if (typeof window !== 'undefined' && (window as any).VerbaNative?.speak) {
      (window as any).VerbaNative.speak(text, lang);
      return;
    }
    if (!('speechSynthesis' in window)) return;
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

  const stopListening = useCallback(() => {
    if (typeof window !== 'undefined' && (window as any).VerbaNative?.stopListening) {
      (window as any).VerbaNative.stopListening();
    }
    rec.current?.stop();
    rec.current = null;
    setListening(false);
  }, []);

  const listen = useCallback((lang: Lang, onText: (t: string, final: boolean) => void, onError?: (msg: string) => void) => {
    stopListening();
    if (typeof window !== 'undefined' && (window as any).VerbaNative?.startListening) {
      (window as any).onVerbaSpeechResult = (txt: string, final: boolean) => {
        onText(txt, final);
        if (final) setListening(false);
      };
      (window as any).onVerbaSpeechError = (errMsg: string) => {
        onError?.(errMsg);
        setListening(false);
      };
      (window as any).onVerbaSpeechEnd = () => {
        setListening(false);
      };
      (window as any).VerbaNative.startListening(lang);
      setListening(true);
      return;
    }

    const Ctor: SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!Ctor) { onError?.('Speech recognition is not supported in this browser. Please type.'); return; }
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

  useEffect(() => () => {
    if (typeof window !== 'undefined' && (window as any).VerbaNative) {
      (window as any).VerbaNative.stopListening?.();
      (window as any).VerbaNative.stopSpeaking?.();
    }
    rec.current?.stop();
    window.speechSynthesis?.cancel();
  }, []);

  return { speak, speaking, listen, stopListening, listening, supported };
}
