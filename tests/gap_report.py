import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from nlp_pipeline import EnglishToSignPipeline

pipe = EnglishToSignPipeline()

phrases = [
    # Banking / ATM
    "Please enter your PIN",
    "Sign here on the form",
    "Please show your ID proof",
    "Insert your chip card",
    "How much do you want to withdraw?",
    "Please collect your receipt",
    # Hospital / Medical
    "Where does it hurt?",
    "Since how many days?",
    "Please show your insurance card",
    "The doctor will see you now",
    "Please take a seat",
    "Do you have any allergy?",
]

print("================================================================================")
print("=== VERBA QUICK PHRASES ADDENDUM — REGISTRY GAP REPORT (BEFORE MODIFICATION) ===")
print("================================================================================")

for idx, p in enumerate(phrases, 1):
    res = pipe.translate(p)
    domain = "Banking / ATM" if idx <= 6 else "Hospital / Medical"
    print(f"\nPhrase {idx} ({domain}): '{p}'")
    print(f"  Normalized:         {res['normalized_text']}")
    print(f"  Translation Status: {res['translation_status']}")
    print(f"  Intent / Q-Type:    {res['semantic'].get('intent')} / {res['semantic'].get('question_type')}")
    
    resolved_signs = []
    missing_signs = []
    for s in res["isl"].get("signs", []):
        mtype = s.get("match_type", "unknown")
        gloss = s["gloss"]
        role = s.get("role", "concept")
        if mtype != "unsupported":
            resolved_signs.append(f"{gloss} ({mtype}, role={role})")
        else:
            missing_signs.append(f"{gloss} (unsupported, role={role})")
            
    print(f"  Resolved Concepts:  {resolved_signs if resolved_signs else 'None'}")
    print(f"  Missing Concepts:   {res['missing_concepts']}")
    print(f"  Video Assets:       {[a['gloss'] for a in res['realization']['assets']]}")
    print(f"  Fallback Text:      {res['fallback_text']}")
