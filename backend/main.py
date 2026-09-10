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
import json, os, time
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator
from gloss import to_isl_gloss, build_sentence, lookup_assets

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "..", "assets")
VOCAB_DIR = os.path.join(ROOT, "vocab")
FRAME_DIM = 270

app = FastAPI(title="Vebra")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if os.path.isdir(ASSETS):
    app.mount("/media", StaticFiles(directory=ASSETS), name="media")  # NOT /assets: the built frontend already uses /assets/*


def load_vocab(name: str) -> dict:
    path = os.path.join(VOCAB_DIR, f"{name}.json")
    if not os.path.exists(path):
        name, path = "hospital", os.path.join(VOCAB_DIR, "hospital.json")
    v = json.load(open(path, encoding="utf-8"))
    v["all"] = v["core"] + v["glosses"]
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
    return {"ok": True, "model": RECOGNIZER_OK, "scenarios": sorted(f[:-5] for f in os.listdir(VOCAB_DIR) if f.endswith(".json"))}


@app.get("/vocab/{name}")
def vocab(name: str):
    v = load_vocab(name)
    return JSONResponse({k: v[k] for k in ("scenario", "core", "glosses", "presets", "theme")})


@app.websocket("/ws/session")
async def session(ws: WebSocket, scenario: str = Query("hospital")):
    await ws.accept()
    vocab = load_vocab(scenario)
    index = load_index()
    rec = SignRecognizer() if RECOGNIZER_OK else None
    await ws.send_json({"type": "hello", "model": RECOGNIZER_OK, "scenario": vocab["scenario"],
                        "vocab": vocab["all"], "presets": vocab["presets"], "clips": sorted(index)})
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
                text = str(msg.get("text", ""))[:300]
                lang = msg.get("lang", "en")
                glosses = to_isl_gloss(text, set(vocab["all"]))
                items, fs = lookup_assets(glosses, index)
                await ws.send_json({"type": "sign_sequence", "text": text, "lang": lang,
                                    "glosses": glosses, "items": items, "fingerspell": fs})

            elif t == "build_sentence":
                glosses = [str(g) for g in msg.get("glosses", [])][:12]
                lang = msg.get("lang", "en")
                text, source = build_sentence(glosses, lang)
                await ws.send_json({"type": "sentence", "text": text, "source": source, "glosses": glosses})

            elif t == "scenario":
                vocab = load_vocab(str(msg.get("value", "hospital")))
                if rec: rec.reset()
                await ws.send_json({"type": "hello", "model": RECOGNIZER_OK, "scenario": vocab["scenario"],
                                    "vocab": vocab["all"], "presets": vocab["presets"], "clips": sorted(index)})

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
