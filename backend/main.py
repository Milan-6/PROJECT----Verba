"""
Vebra backend — FastAPI + native WebSocket.

  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

WebSocket  /ws/session?scenario=hospital
  client -> server : frame {v:[270]} | text_in {text,lang} | scenario {value} | build_sentence {glosses,lang} | reset
  server -> client : hello {model, vocab} | candidate {gloss,confidence} | idle | sign_sequence {...} | sentence {...} | error
Only 270-float landmark vectors are ever received. No video, no audio, no storage.
Runs without a trained model (recognition disabled, everything else works) so the UI can be
built and demoed before data collection is finished.
"""
from __future__ import annotations
import json, os, time, sys
from typing import Any

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator
from gloss import to_isl_gloss, build_sentence, lookup_assets
from nlp_pipeline import EnglishToSignPipeline, SignAssetRegistry, is_sentence_fully_supported
from scenario_routing import (
    resolve_scenario_requirements,
    resolve_dataset_for_scenario,
    generate_scenario_manifest,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "..", "assets")
VOCAB_DIR = os.path.join(ROOT, "vocab")
FRAME_DIM = 270

app = FastAPI(title="Vebra")

# Explicit CORS configuration: Capacitor mobile app, Vite dev, and local network IPs
ALLOWED_ORIGINS = [
    "http://localhost",
    "https://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "capacitor://localhost",
    "ionic://localhost",
]
LAN_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=LAN_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if os.path.isdir(ASSETS):
    app.mount("/media", StaticFiles(directory=ASSETS), name="media")  # NOT /assets: the built frontend already uses /assets/*

# ---- NLP Pipeline & Sign Registry ----
nlp_pipe = EnglishToSignPipeline()


def load_vocab(name: str) -> dict:
    path = os.path.join(VOCAB_DIR, f"{name}.json")
    if not os.path.exists(path):
        name, path = "hospital", os.path.join(VOCAB_DIR, "hospital.json")
    v = json.load(open(path, encoding="utf-8"))
    v["all"] = v["core"] + v["glosses"]
    # Video-gating: Only keep sentences where 100% of glosses resolve to physical video assets
    raw_presets = v.get("presets", [])
    v["presets"] = [p for p in raw_presets if is_sentence_fully_supported(p, scenario=name, pipeline=nlp_pipe)]
    raw_quick = v.get("quick_phrases", [])
    v["quick_phrases"] = [q for q in raw_quick if is_sentence_fully_supported(q, scenario=name, pipeline=nlp_pipe)]
    return v


def load_index() -> dict[str, str]:
    p = os.path.join(ASSETS, "isl", "index.json")
    if not os.path.exists(p):
        return {}
    idx = json.load(open(p, encoding="utf-8"))
    # keep only clips that actually exist so the client never gets a 404
    return {g: f"/media/isl/{c}" for g, c in idx.items() if os.path.exists(os.path.join(ASSETS, "isl", c))}


# ---- model (optional) --------------------------------------------------------
RECOGNIZER_OK = False
try:
    from inference import SignRecognizer  # noqa
    SignRecognizer()  # probe once at startup
    RECOGNIZER_OK = True
    print("[model] loaded backend/models/model.onnx")
except Exception as e:  # missing model or onnxruntime
    print(f"[model] recognition disabled: {e}")


class EnglishToSignRequest(BaseModel):
    text: str
    scenario: str | None = None


class FrameMsg(BaseModel):
    v: list[float]

    @field_validator("v")
    @classmethod
    def _len(cls, v):
        if len(v) != FRAME_DIM:
            raise ValueError(f"expected {FRAME_DIM} floats, got {len(v)}")
        return v


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": RECOGNIZER_OK,
        "scenarios": sorted(f[:-5] for f in os.listdir(VOCAB_DIR) if f.endswith(".json")),
        "registry_signs": len(nlp_pipe.registry.all_signs()),
    }


@app.get("/api/signs")
def list_signs():
    return JSONResponse({
        "count": len(nlp_pipe.registry.all_signs()),
        "signs": nlp_pipe.registry.all_signs(),
        "registry": nlp_pipe.registry._registry,
    })


@app.get("/api/scenarios/manifest")
def scenario_manifest():
    return JSONResponse(generate_scenario_manifest(VOCAB_DIR))


@app.get("/api/scenarios/{name}/requirements")
def scenario_requirements(name: str):
    try:
        return JSONResponse(resolve_scenario_requirements(name, VOCAB_DIR))
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": f"Scenario '{name}' not found"})


@app.get("/api/scenarios/{name}/dataset-plan")
def scenario_dataset_plan(name: str):
    try:
        return JSONResponse(resolve_dataset_for_scenario(name, VOCAB_DIR))
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": f"Scenario '{name}' not found"})


@app.post("/api/english-to-sign")
def english_to_sign(req: EnglishToSignRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text cannot be empty"})
    if len(text) > 1000:
        return JSONResponse(status_code=400, content={"error": "Text exceeds maximum length of 1000 characters"})
    result = nlp_pipe.translate(text, scenario=req.scenario)
    return JSONResponse(result)


@app.get("/vocab/{name}")
def vocab(name: str):
    v = load_vocab(name)
    return JSONResponse({
        "scenario": v["scenario"],
        "core": v["core"],
        "glosses": v["glosses"],
        "presets": v["presets"],
        "quick_phrases": v.get("quick_phrases", []),
        "theme": v.get("theme", name),
    })


@app.websocket("/ws/session")
async def session(ws: WebSocket, scenario: str = Query("hospital")):
    await ws.accept()
    vocab = load_vocab(scenario)
    index = load_index()
    rec = SignRecognizer() if RECOGNIZER_OK else None
    reqs = resolve_scenario_requirements(vocab["scenario"], VOCAB_DIR)
    await ws.send_json({
        "type": "hello",
        "model": RECOGNIZER_OK,
        "scenario": vocab["scenario"],
        "vocab": vocab["all"],
        "presets": vocab["presets"],
        "quick_phrases": vocab.get("quick_phrases", []),
        "clips": sorted(index),
        "requirements": reqs,
    })
    last_frame_t = 0.0
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "bad json"}); continue
            t = msg.get("type")

            if t == "frame":
                # drop frames if we are behind: recency beats completeness
                now = time.time()
                if now - last_frame_t < 0.03:
                    continue
                last_frame_t = now
                try:
                    fm = FrameMsg(v=msg.get("v", []))
                except ValidationError as e:
                    await ws.send_json({"type": "error", "message": e.errors()[0]["msg"]}); continue
                if rec is None:
                    continue
                r = rec.push(fm.v, now)
                if r is None:
                    continue
                if r[0] == "idle":
                    await ws.send_json({"type": "idle"})
                else:
                    await ws.send_json({"type": "candidate", "gloss": r[1], "confidence": r[2]})

            elif t == "text_in":
                text = str(msg.get("text", ""))[:500]
                lang = msg.get("lang", "en")
                scen = msg.get("scenario", vocab.get("scenario", "hospital"))

                # Full Phase 20 NLP pipeline & pre-validated asset resolution (zero 404s)
                pipe_res = nlp_pipe.translate(text, scenario=scen)

                await ws.send_json({
                    "type": "sign_sequence",
                    "text": text,
                    "lang": lang,
                    "scenario": scen,
                    "glosses": [s["gloss"] for s in pipe_res["isl"].get("signs", [])],
                    "items": pipe_res["items"],
                    "fingerspell": pipe_res["missing_concepts"],
                    "translation_status": pipe_res["translation_status"],
                    "available_signs": pipe_res["available_signs"],
                    "missing_concepts": pipe_res["missing_concepts"],
                    "assets": pipe_res["realization"]["assets"],
                    "semantic": pipe_res["semantic"],
                    "isl": pipe_res["isl"],
                    "fallback_text": pipe_res["fallback_text"],
                })

            elif t == "build_sentence":
                glosses = [str(g) for g in msg.get("glosses", [])][:12]
                lang = msg.get("lang", "en")
                text, source = build_sentence(glosses, lang)
                await ws.send_json({"type": "sentence", "text": text, "source": source, "glosses": glosses})

            elif t == "scenario":
                val = str(msg.get("value", "hospital"))
                vocab = load_vocab(val)
                if rec:
                    rec.reset()
                reqs = resolve_scenario_requirements(vocab["scenario"], VOCAB_DIR)
                await ws.send_json({"type": "hello", "model": RECOGNIZER_OK, "scenario": vocab["scenario"],
                                    "vocab": vocab["all"], "presets": vocab["presets"], "clips": sorted(index),
                                    "requirements": reqs})


            elif t == "reset":
                if rec: rec.reset()
                await ws.send_json({"type": "idle"})
            else:
                await ws.send_json({"type": "error", "message": f"unknown type {t}"})
    except WebSocketDisconnect:
        pass


# ---- serve the built frontend (npm run build) from the same port: one process for the demo ----
DIST = os.path.join(ROOT, "..", "frontend", "dist")
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
