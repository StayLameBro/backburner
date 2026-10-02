/* phone-attn.js - phone-attn/phone-attn.h framing + KV store for the wireless
 * WebGPU node. ES module, no dependencies.
 *
 * Wire format (little-endian, both ends; see phone-attn.h):
 *   hdr: u32 magic 'PATN' (0x4E544150) | u32 type | u64 payload_len | payload
 * Types: HELLO=1 HELLO_OK=2 CONFIG=3 APPEND=4 TRUNCATE=5 ATTN=6 ATTN_OK=7
 *        STATS=8 OK=9 ERR=10 BYE=11 PING=12 ATTN_BIG=13
 *
 * Constants (must match phone-attn.h): NR=48 rows per KV head
 * (8 tokens x GQA 6), HD=256 head dim, KEY_ALIGN=64, VERSION=3, PAGE=4096.
 */

export const MAGIC = 0x4e544150;
export const VERSION = 3;
export const NR = 48;
export const HD = 256;
export const KEY_ALIGN = 64;

export const T = {
  HELLO: 1, HELLO_OK: 2, CONFIG: 3, APPEND: 4, TRUNCATE: 5, ATTN: 6,
  ATTN_OK: 7, STATS: 8, OK: 9, ERR: 10, BYE: 11, PING: 12, ATTN_BIG: 13,
};

/* ---- f16 codecs (the wire uses f16 for Q/O; storage here is f16 too) ---- */

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

export function f16ToF32(bits) {
  const s = (bits & 0x8000) << 16;
  let e = (bits >> 10) & 0x1f;
  let m = bits & 0x3ff;
  if (e === 0) {
    if (m === 0) { _u32[0] = s; return _f32[0]; }
    // subnormal -> normalize
    while ((m & 0x400) === 0) { m <<= 1; e -= 1; }
    e += 1; m &= ~0x400;
  } else if (e === 31) {
    _u32[0] = s | 0x7f800000 | (m << 13);
    return _f32[0];
  }
  _u32[0] = s | ((e + 112) << 23) | (m << 13);
  return _f32[0];
}

export function f32ToF16(val) {
  _f32[0] = val;
  const x = _u32[0];
  const s = (x >> 16) & 0x8000;
  const e = ((x >> 23) & 0xff) - 112;
  const m = x & 0x7fffff;
  if (e <= 0) {
    // subnormal / underflow: flush small magnitudes to signed zero (matches
    // the dequant-then-store path; magnitudes here are weight-scale products)
    if (e < -10) return s;
    const shift = 14 - e;
    return s | ((m | 0x800000) >> shift);
  }
  if (e >= 31) return s | 0x7bff; // overflow -> max finite (no inf weights)
  return s | (e << 10) | (m >> 13);
}

/* ---- framing ---- */

export function encodeMsg(type, payload) {
  payload = payload || new Uint8Array(0);
  const out = new Uint8Array(16 + payload.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, MAGIC, true);
  dv.setUint32(4, type, true);
  dv.setBigUint64(8, BigInt(payload.length), true);
  out.set(payload, 16);
  return out;
}

export function decodeHdr(buf, off) {
  const dv = new DataView(buf.buffer, buf.byteOffset + off);
  return {
    magic: dv.getUint32(0, true),
    type: dv.getUint32(4, true),
    len: Number(dv.getBigUint64(8, true)),
  };
}

export function errMsg(text) {
  return encodeMsg(T.ERR, new TextEncoder().encode(text));
}

export function okMsg(extra) {
  return encodeMsg(T.OK, extra || new Uint8Array(0));
}

/* ---- dequant (layouts from pa-tool.cpp: q8_0 = nkv*272B rows of
 * 8 x (f16 d, 32 int8); q4_0 = nkv*144B rows of 8 x (f16 d, 16 nibble
 * bytes, lo=nibble0..15 hi=nibble16..31); f16 = nkv*HD u16s) ---- */

export function rowBytes(isQ8, nkv) {
  if (isQ8 === 0) return nkv * HD * 2;
  if (isQ8 === 1) return nkv * 272;
  return nkv * 144; // q4_0
}

export function headBytes(isQ8) {
  if (isQ8 === 0) return HD * 2;
  if (isQ8 === 1) return 272;
  return 144;
}

/* Dequantize one head-row (Uint8Array view of hb bytes) into out Float32Array
 * of HD, at outOff. dv is a DataView over the same buffer for f16 scales. */
export function dequantHeadRow(isQ8, bytes, byteOff, dv, out, outOff) {
  if (isQ8 === 0) {
    for (let d = 0; d < HD; d++) {
      out[outOff + d] = f16ToF32(dv.getUint16(byteOff + d * 2, true));
    }
    return;
  }
  if (isQ8 === 1) {
    for (let b = 0; b < 8; b++) {
      const d = f16ToF32(dv.getUint16(byteOff + b * 34, true));
      for (let i = 0; i < 32; i++) {
        let q = bytes[byteOff + b * 34 + 2 + i];
        if (q >= 128) q -= 256;
        out[outOff + b * 32 + i] = d * q;
      }
    }
    return;
  }
  // q4_0
  for (let b = 0; b < 8; b++) {
    const d = f16ToF32(dv.getUint16(byteOff + b * 18, true));
    for (let i = 0; i < 16; i++) {
      const q = bytes[byteOff + b * 18 + 2 + i];
      out[outOff + b * 32 + i] = d * ((q & 15) - 8);
      out[outOff + b * 32 + i + 16] = d * ((q >> 4) - 8);
    }
  }
}

/* ---- KV store: per layer, per head, f16-bit storage (iOS store_f16 analog:
 * 2x memory for direct accelerator reads; here for GPU upload + tab memory) */

export class KVStore {
  constructor(capKeys) {
    this.cap = capKeys; // max keys per layer (ERR beyond, like the jetsam guard)
    this.cfg = null;
    this.layers = []; // [{ n, K: Uint16Array(nkv*cap*HD), V: same }]
    this.appended = 0;
  }

  configure(cfg) {
    if (!cfg.n_head_kv || cfg.n_head_kv > 16 || !cfg.n_layer || cfg.n_layer > 256) {
      return 'bad CONFIG';
    }
    this.cfg = cfg;
    this.layers = [];
    for (let l = 0; l < cfg.n_layer; l++) {
      const size = cfg.n_head_kv * this.cap * HD;
      this.layers.push({ n: 0, K: new Uint16Array(size), V: new Uint16Array(size) });
    }
    this.appended = 0;
    return '';
  }

  append(layer, pos0, n, kvBytes) {
    const cfg = this.cfg;
    if (!cfg || layer >= this.layers.length) return { err: 'bad APPEND' };
    const L = this.layers[layer];
    if (pos0 !== L.n) return { err: `APPEND pos0 ${pos0} != held ${L.n}` };
    if (L.n + n > this.cap) return { err: `out of memory at ${L.n} keys` };
    const rs = cfg.rs, hb = headBytes(cfg.is_q8);
    const dv = new DataView(kvBytes.buffer, kvBytes.byteOffset);
    const K = kvBytes.subarray(0, n * rs);
    const V = kvBytes.subarray(n * rs, 2 * n * rs);
    const tmp = new Float32Array(HD);
    for (let j = 0; j < n; j++) {
      for (let h = 0; h < cfg.n_head_kv; h++) {
        const base = (h * this.cap + (L.n + j)) * HD;
        dequantHeadRow(cfg.is_q8, K, j * rs + h * hb, dv, tmp, 0);
        for (let d = 0; d < HD; d++) L.K[base + d] = f32ToF16(tmp[d]);
        const dvV = new DataView(V.buffer, V.byteOffset);
        dequantHeadRow(cfg.is_q8, V, j * rs + h * hb, dvV, tmp, 0);
        for (let d = 0; d < HD; d++) L.V[base + d] = f32ToF16(tmp[d]);
      }
    }
    L.n += n;
    this.appended += n;
    const now = new Uint8Array(4);
    new DataView(now.buffer).setUint32(0, L.n, true);
    return { ok: now };
  }

  truncate(n) {
    for (const L of this.layers) {
      if (L.n > n) L.n = n;
    }
  }
}

/* log-sum-exp merge of partial (O2,m2,l2) into (O,m,l), HD-wide rows.
 * Same math as pa::merge_partial in phone-attn.h. */
export function mergePartial(O, m, l, O2, m2, l2, rows) {
  for (let r = 0; r < rows; r++) {
    if (!(l2[r] > 0)) continue;
    if (!(l[r] > 0)) {
      for (let d = 0; d < HD; d++) O[r * HD + d] = O2[r * HD + d];
      m[r] = m2[r]; l[r] = l2[r];
      continue;
    }
    const M = Math.max(m[r], m2[r]);
    const a = Math.exp(m[r] - M), b = Math.exp(m2[r] - M);
    for (let d = 0; d < HD; d++) O[r * HD + d] = O[r * HD + d] * a + O2[r * HD + d] * b;
    m[r] = M; l[r] = l[r] * a + l2[r] * b;
  }
}
