"""
ISL gloss engine.
  to_isl_gloss(text, vocab)       English/Hindi text  -> ISL gloss order (deterministic rules)
  build_sentence(glosses, lang)   confirmed gloss chips -> natural English / Hindi sentence
  lookup_assets(glosses, index)   gloss -> clip path, or fingerspelling letters

ISL grammar used here (simplified, enough for counter conversations):
  time words first · topic/object before verb (SOV) · no articles/copula · politeness last.
Optional LLM (OpenAI-compatible) polishes both directions when LLM_API_KEY is set;
every call has a 2 s timeout and falls back to the rules, so the demo never depends on it.
"""
from __future__ import annotations
import json, os, re, urllib.request

ARTICLES = {"a", "an", "the"}
COPULA = {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "will", "would", "shall", "can", "could", "may"}
DROP_PREP = {"to", "of", "for", "at", "in", "on", "with", "your", "you", "my", "me", "i", "it", "this", "that", "there", "here",
             "any", "some", "our", "how", "much", "many", "want", "since", "where", "what", "when", "which", "take", "have", "has",
             "show", "give", "does", "need", "get", "very", "so", "just", "see", "are", "am"}
TIME_WORDS = {"today", "tomorrow", "yesterday", "now", "later", "morning", "evening", "month", "week", "day", "days", "hour", "hours", "minute", "minutes"}
POLITE = {"please", "thanks", "thank"}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUM_GLOSS = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN"}
LEMMA = {"documents": "document", "days": "day", "hours": "hour", "cheques": "cheque", "forms": "form", "hurts": "hurt",
         "hurt": "pain", "hurting": "pain", "helping": "help", "helped": "help", "showing": "show", "signing": "sign",
         "waiting": "wait", "insert": "insert", "inserted": "insert", "id": "id_proof", "identification": "id_proof",
         "fever": "fever", "doctor": "doctor", "seat": "sit", "sit": "sit", "money": "cash", "pin": "pin",
         "medicines": "medicine", "tablets": "medicine", "tablet": "medicine", "injection": "injection",
         "thank": "thank_you", "thanks": "thank_you", "allergies": "allergy", "allergic": "allergy",
         "withdrawal": "withdraw", "deposit": "deposit", "deposited": "deposit", "kid": "child", "children": "child",
         "old": "elderly", "elder": "elderly", "receipt": "receipt", "signature": "signature", "sign": "signature",
         "hi": "hello", "hey": "hello", "hellos": "hello", "namaste": "hello", "goodbye": "bye", "byebye": "bye",
         "bye": "bye", "cinema": "absolute_cinema"}
# words that map to a gloss even though the literal word differs
SYNONYM = {"insurance": "INSURANCE_CARD", "card": "INSURANCE_CARD", "id_proof": "IDENTIFICATION", "proof": "IDENTIFICATION",
           "prescription": "PRESCRIPTION", "hurt": "PAIN", "pain": "PAIN", "sit": "WAIT", "collect": "RECEIPT",
           "enter": "PASSWORD", "pin": "PASSWORD", "amount": "AMOUNT",
           "hello": "HELLO", "bye": "BYE", "absolute_cinema": "ABSOLUTE_CINEMA"}
VERBS = {"insert", "show", "give", "take", "wait", "sign", "enter", "write", "collect", "withdraw", "deposit", "help",
         "repeat", "admit", "come", "go", "meet", "pay", "bring", "fill"}


PHRASES = {"absolute cinema": "absolute_cinema", "good morning": "hello",
           "good evening": "hello", "good night": "bye", "see you later": "bye"}


def _tokens(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    for phrase, token in PHRASES.items():          # collapse known multi-word phrases first
        t = t.replace(phrase, token)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'(s|re|ll|ve|d|m)\b", "", t)
    t = re.sub(r"[^a-z0-9ऀ-ॿ_ ]+", " ", t)
    return [w for w in t.split() if w]


def to_isl_gloss(text: str, vocab: set[str]) -> list[str]:
    """Rule-based conversion. Returns glosses in ISL order; unknown content words are
    returned lower-case so the client can fingerspell them."""
    toks = _tokens(text)
    time_part, body, verbs, polite, content_unknown = [], [], [], [], []
    is_question = "?" in text or (toks and toks[0] in {"do", "does", "did", "how", "what", "where", "when", "which", "is", "are"})
    for w in toks:
        if w.isdigit():
            n = int(w)
            body.append(NUM_GLOSS.get(n, w)); continue
        if w in NUM_WORDS:
            body.append(NUM_GLOSS[NUM_WORDS[w]]); continue
        if w in ARTICLES or w in COPULA or w in DROP_PREP:
            continue
        if w == "not":
            body.append("NO"); continue
        w = LEMMA.get(w, w)
        if w in POLITE or w == "thank_you":
            polite.append("THANK_YOU" if w in ("thank", "thanks", "thank_you") else "PLEASE"); continue
        if w in TIME_WORDS:
            g = {"day": "DAYS", "days": "DAYS", "hour": "HOURS", "hours": "HOURS", "month": "MONTH", "today": "TODAY"}.get(w, w.upper())
            time_part.append(g if g in vocab else w); continue
        gloss = w.upper()
        if gloss in vocab:
            (verbs if w in VERBS else body).append(gloss); continue
        syn = SYNONYM.get(w)
        if syn and syn in vocab:
            (verbs if w in VERBS else body).append(syn); continue
        if w in VERBS:
            verbs.append(w); continue          # unknown verb -> fingerspell
        if len(w) > 2:
            content_unknown.append(w)
    order = time_part + body + content_unknown + verbs + polite
    # dedupe while keeping order
    seen, out = set(), []
    for g in order:
        if g not in seen:
            seen.add(g); out.append(g)
    if is_question and "REPEAT" not in out:
        out.append("?")
    return out


# ---------------------------------------------------------------- sentence builder
WORD_EN = {"THANK_YOU": "thank you", "ID_PROOF": "ID proof", "INSURANCE_CARD": "insurance card", "IDENTIFICATION": "ID proof",
           "PASSWORD": "PIN", "DAYS": "days", "HOURS": "hours", "ELDERLY": "elderly person", "NONE": "",
           "HELLO": "hello", "BYE": "goodbye", "ABSOLUTE_CINEMA": "absolute cinema"}
WORD_HI = {"YES": "हाँ", "NO": "नहीं", "HELP": "मदद", "PLEASE": "कृपया", "THANK_YOU": "धन्यवाद", "WAIT": "इंतज़ार", "REPEAT": "दोबारा",
           "WRITE": "लिखिए", "NAME": "नाम", "DONE": "हो गया", "EMERGENCY": "आपातकाल", "DOCTOR": "डॉक्टर", "PAIN": "दर्द",
           "FEVER": "बुखार", "STOMACH": "पेट", "HEAD": "सिर", "CHEST": "छाती", "MEDICINE": "दवा", "INJECTION": "इंजेक्शन",
           "BLOOD": "खून", "ALLERGY": "एलर्जी", "DAYS": "दिन", "HOURS": "घंटे", "INSURANCE_CARD": "बीमा कार्ड",
           "PRESCRIPTION": "पर्ची", "ADMIT": "भर्ती", "COUGH": "खांसी", "VOMIT": "उल्टी", "CHILD": "बच्चा", "ELDERLY": "बुज़ुर्ग",
           "APPOINTMENT": "अपॉइंटमेंट", "ACCOUNT": "खाता", "BALANCE": "बैलेंस", "CASH": "नकद", "DEPOSIT": "जमा", "WITHDRAW": "निकासी",
           "CHEQUE": "चेक", "SIGNATURE": "हस्ताक्षर", "IDENTIFICATION": "पहचान पत्र", "PASSWORD": "पिन", "DOCUMENTS": "दस्तावेज़",
           "INTEREST": "ब्याज", "INCOME": "आय", "BILL": "बिल", "COMPLAINT": "शिकायत", "COUNTER": "काउंटर", "REGISTER": "रजिस्टर",
           "INFORMATION": "जानकारी", "ASSISTANCE": "सहायता", "URGENT": "ज़रूरी", "RECEIPT": "रसीद",
           "HELLO": "नमस्ते", "BYE": "अलविदा", "ABSOLUTE_CINEMA": "एब्सोल्यूट सिनेमा",
           "ONE": "एक", "TWO": "दो", "THREE": "तीन", "FOUR": "चार", "FIVE": "पाँच", "SIX": "छह", "SEVEN": "सात", "EIGHT": "आठ", "NINE": "नौ", "TEN": "दस"}
NUM_EN = {v: k for k, v in NUM_GLOSS.items()}
BODY = {"STOMACH", "HEAD", "CHEST"}


def _w(g: str, lang: str) -> str:
    if lang == "hi":
        return WORD_HI.get(g, g.lower().replace("_", " "))
    return WORD_EN.get(g, g.lower().replace("_", " "))


def build_sentence_rules(glosses: list[str], lang: str = "en") -> str:
    g = [x for x in glosses if x and x not in ("NONE", "THUMBS_UP", "DONE")]
    if not g:
        return ""
    if lang == "hi":
        return " ".join(_w(x, "hi") for x in g) + "।"
    s = set(g)
    parts = []
    # duration: <NUM> DAYS/HOURS
    dur = ""
    for i, x in enumerate(g):
        if x in NUM_EN and i + 1 < len(g) and g[i + 1] in ("DAYS", "HOURS"):
            dur = f"for {NUM_EN[x]} {g[i + 1].lower()}"
    if "PAIN" in s:
        where = next((b for b in g if b in BODY), None)
        parts.append(f"I have {_w(where, 'en')} pain" if where else "I have pain")
        if dur: parts[-1] += " " + dur
    elif "FEVER" in s or "COUGH" in s or "VOMIT" in s:
        sym = [x for x in g if x in ("FEVER", "COUGH", "VOMIT")]
        parts.append("I have " + " and ".join(_w(x, "en") for x in sym) + (" " + dur if dur else ""))
    if "ABSOLUTE_CINEMA" in s: parts.append("Absolute cinema")
    if "EMERGENCY" in s: parts.insert(0, "This is an emergency")
    if "HELP" in s and "EMERGENCY" not in s: parts.append("I need help")
    if "DOCTOR" in s: parts.append("I want to see a doctor")
    if "MEDICINE" in s: parts.append("I need medicine")
    if "ALLERGY" in s: parts.append("I have an allergy")
    if "APPOINTMENT" in s: parts.append("I have an appointment")
    if "WITHDRAW" in s:
        amt = next((NUM_EN[x] for x in g if x in NUM_EN), None)
        parts.append(f"I want to withdraw {amt} thousand rupees" if amt and "DAYS" not in s else "I want to withdraw cash")
    if "DEPOSIT" in s: parts.append("I want to deposit " + ("a cheque" if "CHEQUE" in s else "cash"))
    if "BALANCE" in s: parts.append("I want to check my balance")
    if "ACCOUNT" in s and not ({"WITHDRAW", "DEPOSIT", "BALANCE"} & s): parts.append("It is about my account")
    if "COMPLAINT" in s: parts.append("I want to make a complaint")
    if "RECEIPT" in s: parts.append("Please give me a receipt")
    if "REPEAT" in s: parts.append("Please repeat that")
    if "WRITE" in s: parts.append("Please write it down")
    if "WAIT" in s: parts.append("Please wait")
    if "NAME" in s: parts.append("My name is ...")
    if "YES" in s and len(g) == 1: parts.append("Yes")
    if "NO" in s and len(g) == 1: parts.append("No")
    if "THANK_YOU" in s: parts.append("Thank you")
    if "HELLO" in s: parts.insert(0, "Hello")          # greeting opens the sentence
    if "BYE" in s: parts.append("Goodbye")             # farewell closes it
    used = {"PAIN", "FEVER", "COUGH", "VOMIT", "EMERGENCY", "HELP", "DOCTOR", "MEDICINE", "ALLERGY", "APPOINTMENT", "WITHDRAW",
            "DEPOSIT", "BALANCE", "ACCOUNT", "COMPLAINT", "RECEIPT", "REPEAT", "WRITE", "WAIT", "NAME", "THANK_YOU", "CHEQUE",
            "DAYS", "HOURS", "PLEASE", "HELLO", "BYE", "ABSOLUTE_CINEMA"} | BODY | set(NUM_EN) | ({"YES", "NO"} if len(g) == 1 else set())
    rest = [x for x in g if x not in used]
    if rest:
        parts.append(", ".join(_w(x, "en") for x in rest))
    if not parts:
        parts.append(" ".join(_w(x, "en") for x in g))
    out = ". ".join(p.strip() for p in parts if p)
    if "PLEASE" in s and not out.lower().startswith("please"):
        out += ", please"
    return out[0].upper() + out[1:] + "."


# ---------------------------------------------------------------- optional LLM
def _llm(prompt: str, timeout: float = 2.0) -> str | None:
    key = os.getenv("LLM_API_KEY")
    if not key:
        return None
    url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
    body = json.dumps({"model": os.getenv("LLM_MODEL", "gpt-4o-mini"), "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def build_sentence(glosses: list[str], lang: str = "en") -> tuple[str, str]:
    rules = build_sentence_rules(glosses, lang)
    out = _llm(f"A Deaf person signed these Indian Sign Language glosses in order: {' '.join(glosses)}. "
               f"Write ONE short natural first-person sentence in {'Hindi' if lang == 'hi' else 'English'} that says "
               f"exactly this and nothing more. Do not add facts. Reply with the sentence only.")
    if out and 0 < len(out) < 160:
        return out, "llm"
    return rules, "rules"


def lookup_assets(glosses: list[str], index: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Returns (items, fingerspell). items = [{gloss, clip|None, letters|None}]"""
    items, fs = [], []
    for g in glosses:
        if g == "?":
            items.append({"gloss": "?", "clip": None, "letters": None}); continue
        clip = index.get(g) or index.get(g.lower())
        if clip:
            items.append({"gloss": g, "clip": clip, "letters": None})
        else:
            letters = [c for c in g.replace("_", " ").lower() if c.isalpha() or c == " "]
            items.append({"gloss": g, "clip": None, "letters": letters}); fs.append(g)
    return items, fs
