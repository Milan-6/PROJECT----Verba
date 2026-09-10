// Downloads the two MediaPipe model files into public/mediapipe so the app runs with no internet.
import { writeFile, mkdir } from 'node:fs/promises';
const files = {
  'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  'pose_landmarker_lite.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
};
await mkdir('public/mediapipe', { recursive: true });
for (const [name, url] of Object.entries(files)) {
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`);
  await writeFile(`public/mediapipe/${name}`, Buffer.from(await r.arrayBuffer()));
  console.log('saved public/mediapipe/' + name);
}
