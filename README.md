# Vebra

**Two-way Indian Sign Language communication assistant for hospital receptions and bank counters.**
JOYXoR 2K26 · Problem Statement 5 · communication assistant, not a translator.

```
Deaf person  ── camera ──► MediaPipe (in browser) ──► 270 landmark numbers ──► MLP ──► candidate sign
                                                                         signer confirms ✓ ──► sentence ──► spoken aloud
Hearing person ── types / speaks ──► ISL gloss order ──► big text + ISL clips (or fingerspelling) ──► Deaf person
```

Only landmark numbers ever leave the browser. No video, no audio, nothing stored.

---

## Run it (3 commands)

```bash
# 1. backend + built frontend on one port
./run.sh                  # Windows: run.bat        -> http://localhost:8000
# or explicitly:  cd backend && python -m uvicorn main:app --port 8010

# 2. (optional, once, needs internet) bundle MediaPipe models so the demo works with no Wi-Fi
cd frontend && npm install && npm run models && npm run build

# 3. development mode with hot reload (two terminals)
cd backend  && uvicorn main:app --reload --port 8000
cd frontend && npm run dev                            -> http://localhost:5173
```

The app runs **before you train anything**: hearing → Deaf works fully; the sign-recognition side shows a yellow banner until `backend/models/model.onnx` exists.

---

## Train the recognizer (do this in one sitting, all teammates present)

```bash
cd ml_pipeline
pip install -r requirements.txt

# record: every gloss × every teammate × ~35 samples, at 1 m and 2 m, two lightings
python collect_data.py --scenario hospital --signer bhargav --samples 35
python collect_data.py --scenario bank     --signer bhargav --samples 35
python collect_data.py --gloss NONE        --signer bhargav --samples 80     # idle: hands down, scratching nose, reaching
#   ...repeat for priya, rahul, ...   SPACE = record   s = skip gloss   q = quit
#   start AND end every sample in the rest position. Both hands in frame. Plain background.

# optional: bootstrap from INCLUDE (IIT-Madras ISL dataset, the thesis "General Dataset")
python preprocess_video.py --root ~/INCLUDE --map include_map.json --signer include

# train — hold out one teammate you never trained on
python train_mlp.py --holdout priya --epochs 40 --aug 4
#   prints random-split acc (optimistic) AND held-out acc (honest). Report the honest one.
#   writes backend/models/{model.onnx, labels.json, scaler.json, report.json}

# evaluate the way the demo will run
python evaluate.py --holdout priya
#   [idle test]   N NONE sequences -> 0 false candidates      <- must be 0
#   [live-style]  first candidate == label: x/y               <- target >= 0.90
#   weakest classes -> record 20 more of those, retrain

# restart the backend; the banner disappears and the camera panel starts recognising
```

### Method (from the IIT-Madras thesis *Sign Language Translation*, Sridhar 2019)

| Thesis | Vebra |
|---|---|
| Skeletal keypoints per frame (OpenPose), face ignored | Hands 21×2 + upper-body pose (MediaPipe, in-browser) |
| **Limbs** = vectors between adjacent joints: 4 arm limbs, 4 per finger | Same: 4 arm limbs (2-D) + 40 finger limbs (3-D) |
| Normalise by a body length (eye distance) | Shoulder width (steadier at 1–2 m), eye-distance fallback |
| Limb **velocity** and **acceleration** | Same, summarised as mean/std over a 30-frame window |
| Few clips per class → engineer features, don't learn raw sequences | Same → 1 322 window statistics into a small MLP |
| 57-word banking dataset chosen with interpreters | Bank vocabulary uses 16 of those words + counter basics |

Added: body-relative *location* (ISL depends on where a sign is made), an explicit `NONE` class, mirror augmentation for left-handed signers, shift-pad augmentation so sliding windows that are half sign / half rest are recognised, signer-held-out validation, and a confirm-before-speak UI.

---

## How the demo protects itself

* **Candidate → confirm → speak.** A recognised sign is only a *candidate* (prob > 0.75 for 2 consecutive windows). The signer taps ✓ or signs THUMBS_UP. Nothing is spoken until they press Speak and approve the sentence. A misrecognition is a tap, not an embarrassment.
* **`NONE` class + idle test.** `evaluate.py` streams your idle recordings through the real commit logic and must report 0 false fires.
* **Offline by design.** MediaPipe WASM is bundled; `npm run models` bundles the model files; sentence building and gloss ordering are rule-based (an LLM only polishes when `LLM_API_KEY` is set, 2 s timeout). Web Speech STT needs internet — typing always works.
* **Bank mode:** the camera pauses while the PIN field is focused; "Clear session" wipes everything.
* **Hospital mode:** the EMERGENCY button speaks immediately, bypassing confirmation.

## Why the old camera froze after one second
`startCamera` depended on `selectedCamera` and also called `setSelectedCamera()` → React rebuilt the callback → effect re-ran → stop/start loop. `useCamera.ts` keeps the device in a ref, never auto-switches, releases the old stream first, and guards double-starts. The other "loading forever" cause is MediaPipe's first download (~10 MB); the panel now says "Loading hand model…" and the WASM is served locally.

---

## Layout

```
run.sh / run.bat                 one-command demo (API + built UI on :8000)
backend/main.py                  FastAPI, /ws/session, /health, /vocab/{name}, serves /assets and frontend/dist
backend/inference.py             SignRecognizer: ring buffer, ONNX, commit logic
backend/gloss.py                 text -> ISL gloss order; gloss chips -> English/Hindi sentence; clip lookup
backend/vocab/{hospital,bank}.json
ml_pipeline/features.py          270-dim frame vector, 1322-dim window features, augmentation (mirror of features.ts)
ml_pipeline/mp_extract.py        MediaPipe Tasks wrapper (downloads .task models on first run)
ml_pipeline/collect_data.py      webcam -> data/<GLOSS>/<signer>_<n>.npy (30, 270)
ml_pipeline/preprocess_video.py  INCLUDE / phone clips -> same format
ml_pipeline/train_mlp.py         PyTorch MLP, signer-held-out validation, ONNX export + verification
ml_pipeline/evaluate.py          idle false-fire test, per-sign accuracy, writes redo.bat for weak signs
ml_pipeline/calibrate.py         sweeps the confidence threshold on YOUR data -> backend/models/commit.json
frontend/src/styles.css          THE design system: every colour, type, motion token lives here
frontend/src/App.tsx             two-panel app: I sign / I speak-type, scenario + language switch
frontend/src/components/         CameraPanel, CandidateCard, SentenceStrip, HearingInput, SignPlayer, TopBar
frontend/src/hooks/              useCamera, useLandmarks, useSocket, useSpeech
frontend/src/lib/features.ts     exact JS mirror of features.py (verified to 1e-7)
frontend/public/mediapipe/       bundled WASM (+ models after `npm run models`)
assets/isl/index.json            GLOSS -> clip file (served at /media/isl/…); record 1-2 s clips and list them here
```

## WebSocket protocol — `/ws/session?scenario=hospital|bank`

```
client → server   frame {v:[270]} · text_in {text,lang} · build_sentence {glosses,lang} · scenario {value} · reset
server → client   hello {model,vocab,presets,clips} · candidate {gloss,confidence} · idle
                  sign_sequence {text,glosses,items:[{gloss,clip|null,letters|null}]} · sentence {text,source} · error
```

## Measured results (first trained model, 10 Sep 2026)

Recorded by one signer in ~15 minutes: 306 sequences across 7 classes
(NONE 120, PAIN 33, FEVER 32, TWO 31, DAYS 30, YES 30, NO 30).
The vocabulary has since grown past 12 classes — see `backend/models/report.json` for the
current run, and `weak.json` for per-sign live accuracy.

| Metric | Result | Why it matters |
|---|---|---|
| Idle false-fires | **0 / 120 NONE sequences** | The app never blurts a word while nobody is signing — the failure that kills live demos |
| Live-style accuracy | **168 / 186 = 90.3 %** | Streamed through the real sliding-window commit logic, not a static test set |
| Nothing fired | 15 / 186 | The *safe* failure: the signer just repeats the sign, nothing wrong is spoken |
| Wrong word fired | 3 / 186 | Almost entirely TWO → DAYS (shared handshape, separated only by motion) |
| Per-class | DAYS 0.97 · PAIN 0.94 · YES 0.93 · NO 0.93 · FEVER 0.91 · TWO 0.74 | TWO is the one to improve with 20 more samples |
| ONNX vs PyTorch | max diff 9.5e-07 | Export is faithful |
| Inference | 0.03 ms / sample, CPU | No GPU needed at the venue |

**Read this honestly:** one signer, so a random split lets the model see other samples of the
same hands — the number is optimistic. Record a teammate (`--signer priya`) and retrain with
`--holdout priya` to get the figure you can defend when a judge asks "does it work for anyone else?"

## Also verified
TypeScript compiles clean · Vite production build OK · backend WebSocket exercised for every message type · JS/Python feature vectors identical (max diff 2e-7) · UI driven headless in Chromium: presets, scenario switch, Hindi toggle, PIN pause · running end-to-end on Windows with `/health` reporting `model: true`.

## Honest limits
~35–40 signs per scenario, recognised as isolated signs with the signer confirming each one. No continuous signing, no facial grammar, no regional ISL variants. Real deployment needs co-design with Deaf users and ISL interpreters.


---

## Design system

`frontend/src/styles.css` is the single source of visual truth. An eleven-step neutral ramp
(pure white -> charcoal -> true black), a type scale with all headings explicitly bold, spacing,
radii, shadows, and one easing curve with three durations. Components consume tokens only, so the
whole product re-skins from that one block.

Four things stay chromatic on purpose, because they carry meaning rather than style: the
hand-tracking chip (green / amber / grey), the camera-error overlay, the connection dot, and the
EMERGENCY button. Every animation is wrapped in `prefers-reduced-motion: reduce` — a Deaf user is
visually scanning for the recognised word, and motion competes for that attention.
#   P R O J E C T - - - - V e r b a  
 