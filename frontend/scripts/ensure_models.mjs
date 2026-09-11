// Ensures both MediaPipe model files exist in public/mediapipe before building
import { stat, copyFile, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const publicDir = resolve(__dirname, '../public/mediapipe');
const mlDir = resolve(__dirname, '../../ml_pipeline/models');

const models = {
  'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  'pose_landmarker_lite.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
};

await mkdir(publicDir, { recursive: true });

for (const [name, url] of Object.entries(models)) {
  const target = resolve(publicDir, name);
  let valid = false;
  if (existsSync(target)) {
    const s = await stat(target);
    if (s.size > 1024 * 1024) valid = true;
  }
  if (valid) {
    console.log(`[ensure_models] Already present: public/mediapipe/${name}`);
    continue;
  }

  // Check ml_pipeline/models
  const localCopy = resolve(mlDir, name);
  if (existsSync(localCopy)) {
    const s = await stat(localCopy);
    if (s.size > 1024 * 1024) {
      await copyFile(localCopy, target);
      console.log(`[ensure_models] Copied from ml_pipeline/models: public/mediapipe/${name}`);
      continue;
    }
  }

  // Fallback: download from GCS
  console.log(`[ensure_models] Downloading from CDN: ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download ${url}: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(target, buf);
  console.log(`[ensure_models] Downloaded and saved: public/mediapipe/${name} (${(buf.length / 1024 / 1024).toFixed(2)} MB)`);
}
