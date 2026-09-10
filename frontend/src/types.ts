export type Scenario = 'hospital' | 'bank';
export type Lang = 'en' | 'hi';

export type TranslationStatus = 'supported' | 'partially_supported' | 'unsupported';

export interface SignItem {
  gloss: string;
  clip: string | null;
  duration?: number | null;
  letters: string[] | null;
  role?: string;
}

export type ServerMsg =
  | { type: 'hello'; model: boolean; scenario: Scenario; vocab: string[]; presets: string[]; clips: string[] }
  | { type: 'candidate'; gloss: string; confidence: number }
  | { type: 'idle' }
  | {
      type: 'sign_sequence';
      text: string;
      lang: Lang;
      glosses: string[];
      items: SignItem[];
      fingerspell: string[];
      translation_status?: TranslationStatus;
      available_signs?: string[];
      missing_concepts?: string[];
      fallback_text?: string;
    }
  | { type: 'sentence'; text: string; source: 'rules' | 'llm'; glosses: string[] }
  | { type: 'error'; message: string };

export type ClientMsg =
  | { type: 'frame'; t: number; v: number[] }
  | { type: 'text_in'; text: string; lang: Lang }
  | { type: 'scenario'; value: Scenario }
  | { type: 'build_sentence'; glosses: string[]; lang: Lang }
  | { type: 'reset' };
