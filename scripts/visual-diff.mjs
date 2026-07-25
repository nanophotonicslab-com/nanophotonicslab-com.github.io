/**
 * Compare two directories of Lab screenshots pixel by pixel.
 * Run with: npm run visual:diff -- <dirA> <dirB> [--threshold 0.5] [--tolerance 8]
 *
 * Why not just `cmp` the files: the widgets animate (the spectrum rule under
 * the navbar, canvas draws that finish at slightly different times), so no two
 * runs are byte-identical. Snapshot the SAME code twice first to learn the
 * noise floor of each page, then judge a real change against it. Pages built
 * on litegraph or three.js sit far above the rest and are best eyeballed.
 *
 *   --threshold  percent of differing pixels above which a page is reported
 *                as changed and the process exits non-zero (default 0.5)
 *   --tolerance  per-channel 0-255 difference below which pixels count as
 *                equal, absorbing antialiasing jitter (default 8)
 *
 * Decodes PNG itself (zlib is in the standard library) to avoid a dependency.
 */
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { join } from 'node:path';

/** Minimal 8-bit PNG decoder: enough for Chromium screenshots. */
function decodePNG(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  let off = 8, w = 0, h = 0, colorType = 0;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0);
      h = data.readUInt32BE(4);
      if (data[8] !== 8) throw new Error(`bit depth ${data[8]} unsupported`);
      colorType = data[9];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[colorType];
  if (!channels) throw new Error(`colour type ${colorType} unsupported`);

  const raw = inflateSync(Buffer.concat(idat));
  const stride = w * channels;
  const out = Buffer.alloc(h * stride);
  let pos = 0;
  for (let y = 0; y < h; y++) {
    const filter = raw[pos++];
    const line = raw.subarray(pos, pos + stride);
    pos += stride;
    const cur = out.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? cur[x - channels] : 0;
      const b = prev ? prev[x] : 0;
      const c = prev && x >= channels ? prev[x - channels] : 0;
      let v = line[x];
      switch (filter) {
        case 0: break;
        case 1: v += a; break;
        case 2: v += b; break;
        case 3: v += (a + b) >> 1; break;
        case 4: {
          const p = a + b - c;
          const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
          v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
          break;
        }
        default: throw new Error(`filter ${filter}`);
      }
      cur[x] = v & 0xff;
    }
  }
  return { w, h, channels, data: out };
}

/** Percentage of pixels differing by more than `tol`, plus their bounding box. */
function compare(a, b, tol) {
  if (a.w !== b.w || a.h !== b.h) return { sizeMismatch: true, a: [a.w, a.h], b: [b.w, b.h] };
  let bad = 0, minX = Infinity, minY = Infinity, maxX = -1, maxY = -1;
  for (let y = 0; y < a.h; y++) {
    for (let x = 0; x < a.w; x++) {
      const ia = (y * a.w + x) * a.channels;
      const ib = (y * b.w + x) * b.channels;
      let d = 0;
      for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(a.data[ia + k] - b.data[ib + k]));
      if (d > tol) {
        bad++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  return {
    pct: (100 * bad) / (a.w * a.h),
    box: maxY < 0 ? null : { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 },
  };
}

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i < 0 ? fallback : Number(args[i + 1]);
};
const [dirA, dirB] = args.filter((a) => !a.startsWith('--') && !/^[\d.]+$/.test(a));
const threshold = flag('threshold', 0.5);
const tolerance = flag('tolerance', 8);

if (!dirA || !dirB) {
  console.error('usage: npm run visual:diff -- <dirA> <dirB> [--threshold 0.5] [--tolerance 8]');
  process.exit(2);
}

const rows = [];
for (const f of readdirSync(dirA).filter((x) => x.endsWith('.png'))) {
  const pb = join(dirB, f);
  if (!existsSync(pb)) { rows.push({ f, pct: Infinity, note: 'missing in B' }); continue; }
  try {
    const r = compare(decodePNG(readFileSync(join(dirA, f))), decodePNG(readFileSync(pb)), tolerance);
    if (r.sizeMismatch) { rows.push({ f, pct: Infinity, note: `size ${r.a} vs ${r.b}` }); continue; }
    rows.push({
      f,
      pct: r.pct,
      note: r.box ? `y ${r.box.y}..${r.box.y + r.box.h - 1}  x ${r.box.x}..${r.box.x + r.box.w - 1}` : 'identical',
    });
  } catch (e) {
    rows.push({ f, pct: Infinity, note: `error: ${e.message}` });
  }
}

rows.sort((a, b) => b.pct - a.pct);
for (const r of rows) {
  const pct = Number.isFinite(r.pct) ? `${r.pct.toFixed(3)}%` : '  —   ';
  const mark = r.pct > threshold ? '!' : ' ';
  console.log(`${mark} ${pct.padStart(9)}  ${r.f.replace(/\.png$/, '').padEnd(28)} ${r.note}`);
}

const changed = rows.filter((r) => r.pct > threshold);
console.log(`\n${changed.length} of ${rows.length} pages above ${threshold}% (tolerance ${tolerance}/255)`);
if (changed.length) console.log('Diff the same code against itself first — some pages have a high natural noise floor.');
process.exit(changed.length ? 1 : 0);
