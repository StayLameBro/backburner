/* attention.js - attention over held keys via @huggingface/kernels (WebGPU).
 *
 * Per KV head, per 8-token group (see phone-attn/phone-attn.h ATTN layout):
 *   S = (Q * scale) @ K^T   [48,256] x [256,ck]  (ai.onnx.MatMul)
 *   P = softmax(S) rows       (ai.onnx.Softmax)
 *   O += P @ V               [48,ck] x [ck,256]   (ai.onnx.MatMul)
 * with the log-sum-exp chunk merge from phone-attn.h, so chunking changes
 * speed, not answers. lse = m + log(l) per row is done on the CPU from S
 * (the Softmax kernel returns probabilities only).
 *
 * Q rows r = g*8 + t with t >= n_tok are zero-padded: exact-zero rows are
 * skipped (O=0, lse=-inf), matching the server's padding contract.
 * Falls back to exact CPU loops when WebGPU/kernels are unavailable.
 */

import { getKernel } from '@huggingface/kernels';
import { NR, HD, f16ToF32, f32ToF16, mergePartial } from './phone-attn.js';

const CHUNK = 4096; // keys per WebGPU dispatch (mirrors gpu_chunk tuning)

let _kernels = null; // { matmul, softmax } or { cpu: true }
let _kernelError = '';

export function kernelStatus() {
  if (!_kernels) return 'not loaded';
  if (_kernels.cpu) return 'CPU fallback' + (_kernelError ? ` (${_kernelError})` : '');
  return 'WebGPU (matmul, softmax)';
}

export async function ensureKernels(log) {
  if (_kernels) return _kernels;
  if (!('gpu' in navigator)) {
    _kernelError = 'no navigator.gpu';
    _kernels = { cpu: true };
    log && log('WebGPU unavailable: CPU fallback');
    return _kernels;
  }
  try {
    const matmul = await getKernel('webgpu-kernels/ai.onnx.MatMul', { version: 1 });
    const softmax = await getKernel('webgpu-kernels/ai.onnx.Softmax', { version: 1 });
    _kernels = { matmul, softmax };
    log && log('kernels loaded: ai.onnx.MatMul v1, ai.onnx.Softmax v1');
  } catch (e) {
    _kernelError = String((e && e.message) || e);
    _kernels = { cpu: true };
    log && log('kernel load failed, CPU fallback: ' + _kernelError);
  }
  return _kernels;
}

function isZeroRow(Qw, off) {
  for (let d = 0; d < HD; d++) {
    const w = Qw[off + d];
    if (w !== 0 && w !== 0x8000) return false;
  }
  return true;
}

/* CPU reference path (also the no-WebGPU fallback): streaming online
 * softmax over all nk keys, per live row. Exact; slow for large nk. */
function attnCpu(Qb, liveRows, Kf, Vf, nkv, h, nk, cap, out) {
  // out: { O: Float32Array(48*HD), m, l }
  const { O, m, l } = out;
  for (let r = 0; r < NR; r++) {
    if (!liveRows[r]) continue;
    let mr = -Infinity, lr = 0;
    const acc = new Float32Array(HD);
    for (let j = 0; j < nk; j++) {
      let s = 0;
      const kb = (h * cap + j) * HD;
      for (let d = 0; d < HD; d++) s += Qb[r * HD + d] * f16ToF32(Kf[kb + d]);
      const mNew = Math.max(mr, s);
      const alpha = mr === -Infinity ? 0 : Math.exp(mr - mNew);
      const p = Math.exp(s - mNew);
      lr = lr * alpha + p; mr = mNew;
      const vb = (h * cap + j) * HD;
      for (let d = 0; d < HD; d++) acc[d] = acc[d] * alpha + p * f16ToF32(Vf[vb + d]);
    }
    for (let d = 0; d < HD; d++) O[r * HD + d] = acc[d];
    m[r] = mr; l[r] = lr;
  }
}

export async function computeAttn(store, layer, nTok, nkReq, scale, Qw, ng) {
  const L = store.layers[layer];
  const nkv = store.cfg.n_head_kv;
  const cap = store.cap;
  const nk = nkReq || L.n;
  const qn = nkv * NR * HD;
  const Owords = new Uint16Array(ng * qn);
  const lse = new Float32Array(ng * nkv * NR).fill(-Infinity);
  const t0 = performance.now();

  await ensureKernels(null);
  const useGpu = !_kernels.cpu;

  // Per-group Q as f32, scaled, with live-row mask.
  for (let g = 0; g < ng; g++) {
    const Qg = Qw.subarray(g * qn, (g + 1) * qn);
    const Qb = new Float32Array(qn); // [nkv][48][256]
    const live = new Uint8Array(nkv * NR);
    for (let h = 0; h < nkv; h++) {
      for (let r = 0; r < NR; r++) {
        const off = (h * NR + r) * HD;
        if (isZeroRow(Qg, off)) continue;
        live[h * NR + r] = 1;
        for (let d = 0; d < HD; d++) Qb[off + d] = f16ToF32(Qg[off + d]) * scale;
      }
    }

    for (let h = 0; h < nkv; h++) {
      const O = new Float32Array(NR * HD);
      const m = new Float32Array(NR).fill(-Infinity);
      const l = new Float32Array(NR);
      const qOff = h * NR * HD;
      const Qh = Qb.subarray(qOff, qOff + NR * HD);
      const liveH = live.subarray(h * NR, (h + 1) * NR);

      if (useGpu) {
        const O2 = new Float32Array(NR * HD);
        const m2 = new Float32Array(NR);
        const l2 = new Float32Array(NR);
        for (let c0 = 0; c0 < nk; c0 += CHUNK) {
          const ck = Math.min(CHUNK, nk - c0);
          // Transposed K block [256,ck] + V block [ck,256], f32 scratch.
          const Kt = new Float32Array(HD * ck);
          const Vb = new Float32Array(ck * HD);
          for (let j = 0; j < ck; j++) {
            const kb = (h * cap + c0 + j) * HD;
            for (let d = 0; d < HD; d++) {
              const kv = f16ToF32(L.K[kb + d]);
              Kt[d * ck + j] = kv;
              Vb[j * HD + d] = f16ToF32(L.V[kb + d]);
            }
          }
          const { y: Sav } = await _kernels.matmul({
            a: { data: Qh.slice(), shape: [NR, HD] },
            b: { data: Kt, shape: [HD, ck] },
          });
          const S = Sav.data; // Float32Array [48*ck]
          // lse on CPU from S (Softmax kernel returns probs only)
          for (let r = 0; r < NR; r++) {
            if (!liveH[r]) continue;
            let mx = -Infinity;
            for (let j = 0; j < ck; j++) mx = Math.max(mx, S[r * ck + j]);
            m2[r] = mx;
            let s = 0;
            for (let j = 0; j < ck; j++) s += Math.exp(S[r * ck + j] - mx);
            l2[r] = s;
          }
          const { y: Pav } = await _kernels.softmax(
            { x: { data: S, shape: [NR, ck] } },
            { output: 'gpu' },
          );
          const { y: Oav } = await _kernels.matmul({
            a: Pav, // GPU-resident [48,ck]
            b: { data: Vb, shape: [ck, HD] },
          });
          O2.set(Oav.data);
          mergePartial(O, m, l, O2, m2, l2, NR);
        }
      } else {
        attnCpu(Qh, liveH, L.K, L.V, nkv, h, nk, cap, { O, m, l });
      }

      for (let r = 0; r < NR; r++) {
        const row = (g * nkv + h) * NR + r;
        if (liveH[r] && l[r] > 0) {
          const inv = 1 / l[r];
          for (let d = 0; d < HD; d++) {
            Owords[(g * qn) + (h * NR + r) * HD + d] = f32ToF16(O[r * HD + d] * inv);
          }
          lse[row] = m[r] + Math.log(l[r]);
        } else {
          for (let d = 0; d < HD; d++) Owords[(g * qn) + (h * NR + r) * HD + d] = 0;
          lse[row] = -Infinity;
        }
      }
    }
  }

  return { Owords, lse, ms: performance.now() - t0, gpu: useGpu };
}
