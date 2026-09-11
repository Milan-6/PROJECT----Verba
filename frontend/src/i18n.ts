import type { Lang } from './types';

const en = {
  appName: 'Vebra',
  tagline: 'Indian Sign Language communication assistant',
  hospital: 'Hospital', bank: 'Bank', bus_stop: 'Bus Stop', railway: 'Railway', shopping: 'Shopping', government: 'Government',
  emergency: 'EMERGENCY',
  connected: 'Connected', connecting: 'Connecting…', offline: 'Server offline',
  noModel: 'No trained model yet — recognition is off. Run ml_pipeline/train_mlp.py, then restart the server.',
  deafPanel: 'I sign', hearingPanel: 'I speak / type',
  cameraOn: 'Start camera', cameraOff: 'Stop camera',
  listening: 'Listening for a sign…', candidate: 'Did you sign', add: 'Add', discard: 'Discard',
  sentence: 'Your sentence', speak: 'Speak', clear: 'Clear', confirmSpeak: 'Say this?', yes: 'Yes, say it', edit: 'Edit',
  spoken: 'Said aloud', typeHere: 'Type a message for the Deaf person…', send: 'Send', mic: 'Speak',
  micStop: 'Stop', presets: 'Suggested sentences', quickPhrases: 'Quick Phrases', suggestedSentences: 'Suggested Sentences', shownToDeaf: 'Shown to the Deaf person', glossLabel: 'ISL order',
  replay: 'Replay', slow: 'Slow', noClip: 'no clip yet — fingerspelled', vocabNote: (n: number) => `${n} signs in this mode. Communication assistant, not a translator.`,
  pin: 'PIN (camera pauses while typing)', clearSession: 'Clear session', privacy: 'Only landmark numbers leave this browser. No video or audio is sent or stored.',
  emergencySpoken: 'Emergency. I need help immediately.',
  quality: { good: 'Both hands tracked', 'one-hand': 'One hand tracked', 'no-hands': 'Show your hands', 'too-far': 'Come closer to the camera' },
};
const hi: typeof en = {
  appName: 'Vebra',
  tagline: 'भारतीय सांकेतिक भाषा संचार सहायक',
  hospital: 'अस्पताल', bank: 'बैंक', bus_stop: 'बस स्टॉप', railway: 'रेलवे', shopping: 'खरीदारी', government: 'सरकारी कार्यालय',

  emergency: 'आपातकाल',
  connected: 'जुड़ा हुआ', connecting: 'जुड़ रहा है…', offline: 'सर्वर ऑफ़लाइन',
  noModel: 'अभी कोई प्रशिक्षित मॉडल नहीं — पहचान बंद है। ml_pipeline/train_mlp.py चलाएँ और सर्वर फिर से शुरू करें।',
  deafPanel: 'मैं संकेत करता/करती हूँ', hearingPanel: 'मैं बोलता/टाइप करता हूँ',
  cameraOn: 'कैमरा चालू करें', cameraOff: 'कैमरा बंद करें',
  listening: 'संकेत की प्रतीक्षा…', candidate: 'क्या आपने यह संकेत किया', add: 'जोड़ें', discard: 'हटाएँ',
  sentence: 'आपका वाक्य', speak: 'बोलें', clear: 'साफ़ करें', confirmSpeak: 'यह बोलें?', yes: 'हाँ, बोलो', edit: 'बदलें',
  spoken: 'बोल दिया गया', typeHere: 'बधिर व्यक्ति के लिए संदेश लिखें…', send: 'भेजें', mic: 'बोलें',
  micStop: 'रोकें', presets: 'सुझाए गए वाक्य', quickPhrases: 'त्वरित वाक्यांश', suggestedSentences: 'सुझाए गए वाक्य', shownToDeaf: 'बधिर व्यक्ति को दिखाया गया', glossLabel: 'ISL क्रम',
  replay: 'फिर से', slow: 'धीरे', noClip: 'क्लिप नहीं — अक्षरों से', vocabNote: (n: number) => `इस मोड में ${n} संकेत। यह संचार सहायक है, अनुवादक नहीं।`,
  pin: 'पिन (टाइप करते समय कैमरा रुक जाता है)', clearSession: 'सत्र साफ़ करें', privacy: 'केवल लैंडमार्क संख्याएँ ब्राउज़र से बाहर जाती हैं। कोई वीडियो या ऑडियो नहीं भेजा या रखा जाता।',
  emergencySpoken: 'आपातकाल। मुझे तुरंत मदद चाहिए।',
  quality: { good: 'दोनों हाथ दिख रहे हैं', 'one-hand': 'एक हाथ दिख रहा है', 'no-hands': 'अपने हाथ दिखाएँ', 'too-far': 'कैमरे के पास आएँ' },
};

export const STR: Record<Lang, typeof en> = { en, hi };
