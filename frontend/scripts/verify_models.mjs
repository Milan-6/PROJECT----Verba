// Build-time guard: verifies that dist/mediapipe contains both .task models
import { stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist/mediapipe');

const requiredFiles = ['hand_landmarker.task', 'pose_landmarker_lite.task'];

console.log('[verify_models] Checking build output in dist/mediapipe...');

for (const file of requiredFiles) {
  const filePath = resolve(distDir, file);
  if (!existsSync(filePath)) {
    console.error(`[verify_models] ERROR: Missing ${file} in dist/mediapipe! Build aborted.`);
    process.exit(1);
  }
  const s = await stat(filePath);
  if (s.size < 1024 * 1024) {
    console.error(`[verify_models] ERROR: ${file} in dist/mediapipe is too small (${s.size} bytes)!`);
    process.exit(1);
  }
  console.log(`[verify_models] OK: ${file} (${(s.size / 1024 / 1024).toFixed(2)} MB)`);
}

// Also verify dist/media/isl
const islDist = resolve(__dirname, '../dist/media/isl/clips');
if (!existsSync(islDist)) {
  console.error(`[verify_models] ERROR: Missing ISL clips in dist/media/isl/clips!`);
  process.exit(1);
}
console.log('[verify_models] OK: ISL clips directory verified in dist/media/isl/clips');

console.log('[verify_models] Verification passed: all MediaPipe models and ISL assets packaged successfully.');
