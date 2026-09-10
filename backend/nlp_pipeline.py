"""
VERBA NLP & ISL Pipeline — English Analysis, Semantic Parsing, ISL Transformation, and Asset Resolution.

Main features:
1. English Normalization: Clean whitespace, casing, expand contractions, lemmatize, handle synonyms.
2. Semantic Parser: Extract intent, question type, subject, predicate/action, object, negation, temporal information, politeness.
3. ISL Intermediate Representation (IR): Topic-comment / SOV ordering, temporal-first, negation positioning, question marking, non-manual markers.
4. Sign Asset Resolver: Phrase-level resolution priority, centralized registry lookup, translation status (SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED), missing concept tracking, strict path-traversal security.
"""
from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(ROOT, "sign_registry.json")

# ----------------- Normalization dictionaries & mappings -----------------
CONTRACTIONS = {
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bcan't\b": "cannot",
    r"\bcouldn't\b": "could not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bshouldn't\b": "should not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bi'm\b": "i am",
    r"\byou're\b": "you are",
    r"\bhe's\b": "he is",
    r"\bshe's\b": "she is",
    r"\bit's\b": "it is",
    r"\bwe're\b": "we are",
    r"\bthey're\b": "they are",
    r"\bwhat's\b": "what is",
    r"\bwhere's\b": "where is",
    r"\bthere's\b": "there is",
    r"\bhow's\b": "how is",
    r"\bi've\b": "i have",
    r"\byou've\b": "you have",
    r"\bi'll\b": "i will",
    r"\byou'll\b": "you will",
    r"\blet's\b": "let us",
}

LEMMAS = {
    "helping": "help",
    "helped": "help",
    "waiting": "wait",
    "waited": "wait",
    "seated": "sit",
    "sitting": "sit",
    "eating": "eat",
    "ate": "eat",
    "drinking": "drink",
    "drank": "drink",
    "sleeping": "sleep",
    "slept": "sleep",
    "standing": "stand",
    "stood": "stand",
    "stopping": "stop",
    "stopped": "stop",
    "writing": "write",
    "wrote": "write",
    "thanks": "thank you",
    "thankyou": "thank you",
}

CURATED_SYNONYMS = {
    "hurting": "pain",
    "hurts": "pain",
    "hurt": "pain",
    "ache": "pain",
    "aching": "pain",
    "doc": "doctor",
    "physician": "doctor",
    "tablets": "medicine",
    "tablet": "medicine",
    "pills": "medicine",
    "pill": "medicine",
    "medicines": "medicine",
    "meds": "medicine",
    "drugs": "medicine",
    "id_proof": "identification",
    "proof": "identification",
    "pin": "password",
    "passcode": "password",
    "cash": "cash",
    "rupees": "cash",
    "money": "cash",
    "cheques": "cheque",
    "form": "documents",
    "forms": "documents",
    "papers": "documents",
    "sign": "write",
    "seat": "sit",
    "hi": "hello",
    "hey": "hello",
    "greetings": "hello",
}

SYNONYMS = {**LEMMAS, **CURATED_SYNONYMS}

PHRASE_MAPPINGS = {
    "good morning": "GOOD_MORNING",
    "thank you": "THANK_YOU",
    "thank you very much": "THANK_YOU",
    "how are you": "HOW_ARE_YOU",
    "i am fine": "I_AM_FINE",
    "you are perfect": "YOU_ARE_PERFECT",
    "insurance card": "INSURANCE_CARD",
    "id proof": "ID_PROOF",
    "chip card": "CHIP_CARD",
    "take a seat": "SIT",
    "how much": "HOW_MUCH",
    "how many": "HOW_MANY",
    "after lunch": "AFTER_LUNCH",
    "before food": "BEFORE_FOOD",
    "after food": "AFTER_FOOD",
}

WH_WORDS = {"what", "where", "when", "why", "who", "which", "how", "whose", "whom"}
MODAL_QUESTIONS = {"can", "could", "would", "will", "do", "does", "did", "is", "are", "have", "has", "may"}
POLITE_WORDS = {"please", "kindly"}
TEMPORAL_WORDS = {"today", "tomorrow", "yesterday", "now", "later", "morning", "afternoon", "evening", "night", "days", "hours"}


class NormalizationLayer:
    """Handles case, punctuation, contraction expansion, and synonym unification."""

    @staticmethod
    def normalize(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        # 1. Lowercase and replace unicode quotes
        cleaned = text.lower().strip()
        cleaned = cleaned.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')

        # 2. Expand contractions
        for pattern, replacement in CONTRACTIONS.items():
            cleaned = re.sub(pattern, replacement, cleaned)

        # 3. Clean extra punctuation except terminal question/exclamation markers
        has_question = "?" in cleaned
        cleaned = re.sub(r"[^\w\s\?]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 4. Canonicalize multi-word phrases first
        for phrase, canonical in PHRASE_MAPPINGS.items():
            # word boundary matching
            p_regex = r"\b" + re.escape(phrase) + r"\b"
            cleaned = re.sub(p_regex, canonical.lower(), cleaned)

        # 5. Token-level synonym unification
        tokens = cleaned.split()
        normalized_tokens = []
        for tok in tokens:
            t_clean = tok.rstrip("?")
            syn = SYNONYMS.get(t_clean, t_clean)
            if tok.endswith("?"):
                syn += "?"
            normalized_tokens.append(syn)

        result = " ".join(normalized_tokens)
        if has_question and not result.endswith("?"):
            result += "?"
        return result


class SemanticParser:
    """Extracts structured intent, question type, grammatical roles, and negation."""

    @staticmethod
    def parse(normalized_text: str) -> Dict[str, Any]:
        raw = normalized_text.strip()
        is_q = raw.endswith("?") or any(raw.startswith(w + " ") for w in WH_WORDS | MODAL_QUESTIONS)
        cleaned = raw.rstrip("?").strip()
        tokens = cleaned.split()

        # Intent detection
        intent = "statement"
        q_type = None
        location_query = False

        if any(t in {"hello", "good_morning", "welcome", "bye"} for t in tokens):
            intent = "greeting"
        elif is_q:
            intent = "question"
            if any(w in WH_WORDS for w in tokens) or any(t.startswith("how_") for t in tokens):
                q_type = "wh"
                if "where" in tokens:
                    location_query = True
            else:
                q_type = "yes_no"
        elif any(t in POLITE_WORDS for t in tokens) or (tokens and tokens[0] in {"can", "could", "would", "please"}):
            intent = "request"
        elif tokens and tokens[0] in {"take", "give", "show", "wait", "sit", "write", "stand", "stop"}:
            intent = "command"

        # Check negation
        negation = False
        if any(w in {"not", "no", "never", "don't", "cannot"} for w in tokens):
            negation = True

        # Politeness
        polite = any(w in POLITE_WORDS for w in tokens)

        # Identify subject, action, object
        subject = None
        action = None
        object_ = None
        temporal = None
        detected_phrases = []

        known_actions = {
            "help", "take", "eat", "drink", "sleep", "sit", "stand", "stop", "write", "wait",
            "see", "give", "show", "have", "need", "withdraw", "deposit", "feel", "register"
        }
        known_subjects = {"doctor", "patient", "i", "you", "we", "he", "she", "they", "name"}
        known_objects = {"medicine", "water", "card", "receipt", "document", "pain", "fever", "cash", "pin", "password"}

        GREETINGS_SET = {"hello", "good_morning", "welcome", "bye", "hi", "hey"}
        PRONOUNS_SET = {"i", "you", "me", "we", "us", "he", "she", "they", "them", "my", "your", "our", "their", "his", "her", "this", "that"}
        STOP_TOKENS = WH_WORDS | MODAL_QUESTIONS | GREETINGS_SET | PRONOUNS_SET | POLITE_WORDS | {"the", "a", "an", "is", "am", "are", "was", "were", "do", "does", "did", "to", "of", "for", "in", "on", "not", "no"}

        for t in tokens:
            if t in PHRASE_MAPPINGS.values() or t.upper() in PHRASE_MAPPINGS.values():
                detected_phrases.append(t.upper())
            if t in TEMPORAL_WORDS or t in {"after_lunch", "before_food", "after_food"}:
                temporal = t
            elif t in known_actions and action is None:
                action = t
            elif t in known_subjects and subject is None and t not in GREETINGS_SET:
                subject = t
            elif t in known_objects and object_ is None:
                object_ = t
            elif t not in STOP_TOKENS and t not in known_actions and object_ is None:
                object_ = t

        return {
            "intent": intent,
            "question_type": q_type,
            "location_query": location_query,
            "subject": subject,
            "action": action,
            "object": object_,
            "negation": negation,
            "time": temporal,
            "temporal": temporal,
            "politeness": polite,
            "detected_phrases": detected_phrases,
            "raw_tokens": tokens,
        }


class IslTransformationLayer:
    """Transforms semantic representation into ISL Intermediate Representation (SOV, Time-first)."""

    @staticmethod
    def transform(semantic: Dict[str, Any], normalized_text: str) -> Dict[str, Any]:
        signs = []
        tokens = semantic.get("raw_tokens", [])
        negation = semantic.get("negation", False)
        q_type = semantic.get("question_type")
        added_sign_ids = set()

        def add_sign(sid: str, gloss: str, role: str):
            if sid not in added_sign_ids:
                added_sign_ids.add(sid)
                signs.append({"sign_id": sid, "gloss": gloss, "role": role})

        # 1. Greetings always come first in discourse
        for t in tokens:
            if t in {"hello", "good_morning", "welcome"}:
                add_sign(t.lower(), t.upper(), "greeting")

        # 2. Time/Temporal markers come first in ISL clauses
        if semantic.get("temporal"):
            t = semantic["temporal"]
            add_sign(t.lower(), t.upper(), "time")

        # 3. Topic / Object before Verb (SOV order)
        if semantic.get("object"):
            obj = semantic["object"]
            add_sign(obj.lower(), obj.upper(), "topic")

        # 4. Subject / Actor
        if semantic.get("subject") and semantic["subject"] not in {"i", "you"}:
            subj = semantic["subject"]
            add_sign(subj.lower(), subj.upper(), "subject")

        # 5. Predicate / Action
        if semantic.get("action"):
            act = semantic["action"]
            add_sign(act.lower(), act.upper(), "predicate")

        # If semantic extraction missed explicit key concepts, preserve recognized content tokens
        stopwords = {"the", "a", "an", "is", "am", "are", "was", "were", "do", "does", "did", "can", "could", "would", "will", "i", "you", "me", "your", "my", "to", "of", "for", "in", "on", "not", "no"}
        for t in tokens:
            g = t.upper()
            if t not in stopwords and t not in added_sign_ids:
                add_sign(t.lower(), g, "concept")

        # 6. Negation appears at the end of the predicate clause in ISL
        if negation:
            add_sign("no", "NO", "negation")

        # 7. Politeness (PLEASE / THANK_YOU)
        if semantic.get("politeness"):
            add_sign("please", "PLEASE", "politeness")

        # 8. Questions: Question markers at end in ISL
        if q_type == "wh":
            wh_sign = "where" if semantic.get("location_query") else "what"
            add_sign(wh_sign, wh_sign.upper(), "question")

        return {
            "sentence_type": semantic.get("intent", "statement"),
            "topic": semantic.get("object") or semantic.get("subject"),
            "predicate": semantic.get("action"),
            "negation": negation,
            "question_type": q_type,
            "signs": signs,
            "non_manual_markers": {
                "eyebrows": "furrowed" if q_type == "wh" else "raised" if q_type == "yes_no" else None,
                "head": "shake" if negation else "nod" if semantic.get("intent") == "statement" else None,
            },
            "spatial_reference": None,
        }


class SignAssetRegistry:
    """Centralized registry for approved sign assets with strict path traversal prevention."""

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry_path = registry_path
        self._registry: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
            except Exception as e:
                print(f"[Registry] Error loading {self.registry_path}: {e}")
                self._registry = {}
        else:
            self._registry = {}

    def get(self, sign_id: str) -> Optional[Dict[str, Any]]:
        # Security: reject invalid characters, path traversal
        if not sign_id or not re.match(r"^[a-zA-Z0-9_-]+$", sign_id):
            return None
        sid = sign_id.lower()
        if sid in self._registry:
            return self._registry[sid]
        # Check by gloss
        for entry in self._registry.values():
            if entry.get("gloss", "").lower() == sid:
                return entry
        return None

    def register(self, sign_id: str, gloss: str, asset_type: str, asset: str, verified: bool = True, duration: float = 1.5, **kwargs):
        # Security check
        if not re.match(r"^[a-zA-Z0-9_-]+$", sign_id):
            raise ValueError(f"Invalid sign_id: {sign_id}")
        if ".." in asset or asset.startswith("/") or asset.startswith("\\"):
            raise ValueError(f"Insecure asset path: {asset}")

        sid = sign_id.lower()
        self._registry[sid] = {
            "sign_id": sid,
            "gloss": gloss.upper(),
            "asset_type": asset_type,
            "asset": asset,
            "url": f"/media/isl/{asset}",
            "verified": bool(verified),
            "duration": float(duration),
            **kwargs,
        }
        self.save()

    def save(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2)

    def all_signs(self) -> List[str]:
        return sorted(self._registry.keys())


class EnglishToSignPipeline:
    """Full English -> Normalization -> Semantic IR -> ISL IR -> Verified Video Realization."""

    def __init__(self, registry: Optional[SignAssetRegistry] = None):
        self.registry = registry or SignAssetRegistry()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def translate(self, text: str) -> Dict[str, Any]:
        cache_key = text.strip().lower()
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Fast path: re-verify assets through registry for security
            resolved_assets = []
            available_signs = []
            missing_concepts = []
            for item in cached["isl"].get("signs", []):
                asset_info = self.registry.get(item["sign_id"])
                if asset_info and asset_info.get("verified"):
                    resolved_assets.append({
                        "sign_id": asset_info["sign_id"],
                        "gloss": asset_info["gloss"],
                        "url": asset_info["url"],
                        "duration": asset_info.get("duration", 1.5),
                        "role": item.get("role"),
                    })
                    available_signs.append(asset_info["gloss"])
                else:
                    missing_concepts.append(item["gloss"])
            return {
                **cached,
                "realization": {"renderer": "video", "assets": resolved_assets},
                "available_signs": available_signs,
                "missing_concepts": missing_concepts,
            }

        # 1. Normalization
        normalized = NormalizationLayer.normalize(text)
        if not normalized:
            return {
                "source_text": text,
                "normalized_text": "",
                "translation_status": "unsupported",
                "semantic": {},
                "isl": {"signs": []},
                "realization": {"renderer": "video", "assets": []},
                "available_signs": [],
                "missing_concepts": [],
                "fallback_text": "No text provided.",
            }

        # 2. Semantic Parsing
        semantic = SemanticParser.parse(normalized)

        # 3. ISL IR Transformation
        isl_ir = IslTransformationLayer.transform(semantic, normalized)

        # 4. Realization & Asset Resolution
        resolved_assets = []
        available_signs = []
        missing_concepts = []
        annotated_signs = []
        raw_words = re.findall(r"\w+", text.lower())

        for item in isl_ir.get("signs", []):
            sign_id = item["sign_id"]
            gloss = item["gloss"]

            # Lookup in registry
            asset_info = self.registry.get(sign_id)
            if asset_info and asset_info.get("verified"):
                # Determine match_type tier
                if any(rw in LEMMAS and LEMMAS[rw] == sign_id for rw in raw_words):
                    match_type = "lemma"
                elif any(rw in CURATED_SYNONYMS and CURATED_SYNONYMS[rw] == sign_id for rw in raw_words):
                    match_type = "curated_synonym"
                else:
                    match_type = "exact"

                annotated_signs.append({
                    "sign_id": asset_info["sign_id"],
                    "match_type": match_type,
                    "gloss": asset_info["gloss"],
                    "role": item.get("role"),
                })
                resolved_assets.append({
                    "sign_id": asset_info["sign_id"],
                    "gloss": asset_info["gloss"],
                    "url": asset_info["url"],
                    "duration": asset_info.get("duration", 1.5),
                    "role": item.get("role"),
                })
                available_signs.append(asset_info["gloss"])
            else:
                annotated_signs.append({
                    "sign_id": sign_id,
                    "match_type": "unsupported",
                    "gloss": gloss,
                    "role": item.get("role"),
                })
                missing_concepts.append(gloss)

        isl_ir["signs"] = annotated_signs

        # 5. Translation Status
        total_concepts = len(isl_ir.get("signs", []))
        if len(resolved_assets) == total_concepts and total_concepts > 0:
            status = "supported"
        elif len(resolved_assets) > 0:
            status = "partially_supported"
        else:
            status = "unsupported"

        fallback = text
        if status == "partially_supported":
            fallback = f"Partially translated: {', '.join(available_signs)}. Missing: {', '.join(missing_concepts)}."
        elif status == "unsupported":
            fallback = f"Signs unavailable for: {text}. Showing written text."

        res = {
            "source_text": text,
            "normalized_text": normalized,
            "translation_status": status,
            "semantic": semantic,
            "isl": isl_ir,
            "realization": {
                "renderer": "video",
                "assets": resolved_assets,
            },
            "available_signs": available_signs,
            "missing_concepts": missing_concepts,
            "fallback_text": fallback,
        }
        self._cache[cache_key] = res
        return res
