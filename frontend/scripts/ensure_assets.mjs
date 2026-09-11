// Ensures all ISL video clips and registry index exist in public/media/isl before building
import { cp, mkdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(__dirname, '../../assets/isl');
const targetDir = resolve(__dirname, '../public/media/isl');

console.log('[ensure_assets] Checking ISL video assets in public/media/isl...');

if (!existsSync(srcDir)) {
  console.warn(`[ensure_assets] WARNING: Source ISL directory not found: ${srcDir}`);
  process.exit(0);
}

await mkdir(targetDir, { recursive: true });

try {
  await cp(srcDir, targetDir, { recursive: true });
  console.log(`[ensure_assets] Successfully copied ISL assets from ${srcDir} to ${targetDir}`);
} catch (err) {
  console.error(`[ensure_assets] ERROR copying ISL assets:`, err);
  process.exit(1);
}
