/**
 * Exact mirror of ml_pipeline/features.py::frame_vector. Keep in sync.
 * Layout (FRAME_DIM = 270):
 *   [0:2]     presence flags  left, right
 *   [2:142]   body-normalised L hand 21x3, R hand 21x3, pose 7x2
 *   [142:270] limb vectors    arms 4x2, fingers 40x3
 */
export const FRAME_DIM = 270;
export type Pt = { x: number; y: number; z: number };

const POSE = { NOSE: 0, LEYE: 2, REYE: 5, LSHO: 11, RSHO: 12, LELB: 13, RELB: 14, LWRI: 15, RWRI: 16 };
const POSE_KEEP = [POSE.NOSE, POSE.LSHO, POSE.RSHO, POSE.LELB, POSE.RELB, POSE.LWRI, POSE.RWRI];
const ARM_LIMBS: [number, number][] = [[POSE.LSHO, POSE.LELB], [POSE.LELB, POSE.LWRI], [POSE.RSHO, POSE.RELB], [POSE.RELB, POSE.RWRI]];
const FINGER_LIMBS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]];

const d2 = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const d3 = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

/** Assign detected hands to left/right by proximity to the pose wrists (mirror-safe). */
export function assignHands(hands: Pt[][], pose: Pt[] | null): { left: Pt[] | null; right: Pt[] | null } {
  if (!hands.length) return { left: null, right: null };
  if (!pose) {
    const hs = [...hands].sort((a, b) => a[0].x - b[0].x);
    if (hs.length === 1) return hs[0][0].x < 0.5 ? { left: null, right: hs[0] } : { left: hs[0], right: null };
    return { left: hs[1], right: hs[0] };
  }
  const lw = pose[POSE.LWRI], rw = pose[POSE.RWRI];
  const s = hands.map(h => ({ dl: d2(h[0], lw), dr: d2(h[0], rw), h }));
  if (s.length === 1) return s[0].dl <= s[0].dr ? { left: s[0].h, right: null } : { left: null, right: s[0].h };
  const [a, b] = s;
  return a.dl + b.dr <= a.dr + b.dl ? { left: a.h, right: b.h } : { left: b.h, right: a.h };
}

function bodyScale(pose: Pt[] | null): number {
  if (!pose) return 1;
  const sw = d2(pose[POSE.LSHO], pose[POSE.RSHO]);
  if (sw > 0.02) return sw;
  const ew = d2(pose[POSE.LEYE], pose[POSE.REYE]);
  return ew > 0.005 ? ew * 4 : 1;
}

export function frameVector(left: Pt[] | null, right: Pt[] | null, pose: Pt[] | null): Float32Array {
  const v = new Float32Array(FRAME_DIM);
  v[0] = left ? 1 : 0;
  v[1] = right ? 1 : 0;
  const c = pose ? { x: (pose[POSE.LSHO].x + pose[POSE.RSHO].x) / 2, y: (pose[POSE.LSHO].y + pose[POSE.RSHO].y) / 2 } : { x: 0.5, y: 0.5 };
  const s = bodyScale(pose);
  let o = 2;
  for (const h of [left, right]) {
    if (h) for (let i = 0; i < 21; i++) { v[o + i * 3] = (h[i].x - c.x) / s; v[o + i * 3 + 1] = (h[i].y - c.y) / s; v[o + i * 3 + 2] = h[i].z / s; }
    o += 63;
  }
  if (pose) POSE_KEEP.forEach((k, i) => { v[o + i * 2] = (pose[k].x - c.x) / s; v[o + i * 2 + 1] = (pose[k].y - c.y) / s; });
  o += 14; // o = 142
  if (pose) for (const [a, b] of ARM_LIMBS) { v[o] = (pose[b].x - pose[a].x) / s; v[o + 1] = (pose[b].y - pose[a].y) / s; o += 2; }
  else o += 8;
  for (const h of [left, right]) {
    if (h) {
      const hs = d3(h[9], h[0]) + 1e-6;
      for (const [a, b] of FINGER_LIMBS) { v[o] = (h[b].x - h[a].x) / hs; v[o + 1] = (h[b].y - h[a].y) / hs; v[o + 2] = (h[b].z - h[a].z) / hs; o += 3; }
    } else o += 60;
  }
  if (o !== FRAME_DIM) throw new Error(`feature length ${o} != ${FRAME_DIM}`);
  return v;
}
