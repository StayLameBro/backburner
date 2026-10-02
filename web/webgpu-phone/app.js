/* app.js - wireless WebGPU node runtime.
 *
 * Connects to the host bridge over WebSocket (binary frames carry the exact
 * PATN bytes from phone-attn.h, so the bridge is a dumb pipe). Strict
 * request -> reply, one at a time; messages may arrive coalesced or split,
 * hence the reassembly buffer. Query: ?token= must match the bridge token.
 */

import {
  MAGIC, VERSION, NR, HD, T, encodeMsg, decodeHdr, errMsg, okMsg,
  rowBytes, KVStore,
} from './phone-attn.js';
import { ensureKernels, kernelStatus, computeAttn } from './attention.js';

const $ = (id) => document.getElementById(id);
const logEl = () => $('log');
export function log(s) {
  const el = logEl();
  const line = `${new Date().toLocaleTimeString()} ${s}`;
  el.textContent = (line + '\n' + el.textContent).slice(0, 8000);
}

const store = new KVStore(8192); // keys/layer cap; ERR beyond (jetsam-guard analog)
const stats = { calls: 0, lastMs: 0, gpuMs: 0, startMs: 0 };
let ws = null;
let rxBuf = new Uint8Array(0);
let macPhase = '';

function heldKeys() {
  return store.layers.length ? store.layers[0].n : 0;
}

function refresh() {
  $('held').textContent = heldKeys().toLocaleString();
  $('calls').textContent = String(stats.calls);
  $('lastms').textContent = stats.lastMs ? stats.lastMs.toFixed(1) + ' ms' : '-';
  $('kernels').textContent = kernelStatus();
  $('phase').textContent = macPhase || '-';
}

function send(bytes) {
  ws.send(bytes);
}

async function handleMessage(type, payload) {
  const dv = new DataView(payload.buffer, payload.byteOffset);
  switch (type) {
    case T.HELLO: {
      const rep = new Uint8Array(4 + 4 + 64);
      const rdv = new DataView(rep.buffer);
      rdv.setUint32(0, VERSION, true);
      rdv.setUint32(4, 0, true); // sme2: browser has none; honest
      const dev = `webgpu-phone kernels=matmul,softmax ${navigator.userAgent.slice(0, 40)}`;
      new TextEncoder().encodeInto(dev, new Uint8Array(rep.buffer, 8, 64));
      send(encodeMsg(T.HELLO_OK, rep));
      break;
    }
    case T.CONFIG: {
      if (payload.length < 44) { send(errMsg('short CONFIG')); break; }
      const cfg = {
        n_layer: dv.getUint32(0, true), n_head_kv: dv.getUint32(4, true),
        rs: dv.getUint32(8, true), hb: dv.getUint32(12, true),
        is_q8: dv.getUint32(16, true), sme_workers: dv.getUint32(20, true),
        sme_helpers: dv.getUint32(24, true), gpu_permille: dv.getUint32(28, true),
        gpu_chunk: dv.getUint32(32, true), store_f16: dv.getUint32(36, true),
        gpu_variant: dv.getUint32(40, true),
      };
      if (cfg.rs < cfg.n_head_kv * cfg.hb) { send(errMsg('bad CONFIG')); break; }
      const bad = store.configure(cfg);
      if (bad) { send(errMsg(bad)); break; }
      stats.calls = 0;
      log(`CONFIG: ${cfg.n_layer} layers x ${cfg.n_head_kv} heads, ${['f16', 'q8_0', 'q4_0'][cfg.is_q8] || '?'}`);
      send(okMsg());
      break;
    }
    case T.APPEND: {
      if (payload.length < 12) { send(errMsg('short APPEND')); break; }
      const layer = dv.getUint32(0, true), pos0 = dv.getUint32(4, true), n = dv.getUint32(8, true);
      const rs = store.cfg ? store.cfg.rs : 0;
      if (!store.cfg || payload.length !== 12 + 2 * n * rs) { send(errMsg('bad APPEND')); break; }
      const r = store.append(layer, pos0, n, payload.subarray(12));
      if (r.err) send(errMsg(r.err));
      else send(okMsg(r.ok));
      break;
    }
    case T.TRUNCATE: {
      const n = payload.length >= 4 ? dv.getUint32(0, true) : 0;
      store.truncate(n);
      send(okMsg());
      break;
    }
    case T.ATTN:
    case T.ATTN_BIG: {
      const big = type === T.ATTN_BIG;
      if (payload.length < 12 || !store.cfg) { send(errMsg('bad ATTN')); break; }
      const layer = dv.getUint32(0, true), nTok = dv.getUint32(4, true);
      const nk = dv.getUint32(8, true), scale = dv.getFloat32(12, true);
      const nkv = store.cfg.n_head_kv;
      const qn = nkv * NR * HD;
      const ng = big ? Math.ceil(nTok / 8) : 1;
      if (ng < 1 || ng > 64 || payload.length !== 16 + ng * qn * 2) { send(errMsg('bad ATTN')); break; }
      if (nk > store.layers[layer].n || nk % 1 !== 0) { send(errMsg(`ATTN nk ${nk}`)); break; }
      const Qw = new Uint16Array(payload.buffer, payload.byteOffset + 16, ng * qn);
      const t0 = performance.now();
      const { Owords, lse, gpu } = await computeAttn(store, layer, nTok, nk, scale, Qw, ng);
      const ms = performance.now() - t0;
      stats.calls++; stats.lastMs = ms; stats.gpuMs = gpu ? ms : 0;
      const effNk = nk || store.layers[layer].n;
      const pages = Math.ceil(effNk / 4096);
      // attn_rep: u32 nk; float phone_ms, gpu_ms, sme_ms; u32 gpu_pages, pages
      const rep = new Uint8Array(24);
      const rdv = new DataView(rep.buffer);
      rdv.setUint32(0, effNk, true);
      rdv.setFloat32(4, ms, true);
      rdv.setFloat32(8, gpu ? ms : 0, true);
      rdv.setFloat32(12, 0, true);
      rdv.setUint32(16, 0, true); // gpu_pages: keys run on the WebGPU path
      rdv.setUint32(20, pages, true);
      const head = encodeMsg(T.ATTN_OK, rep);
      const out = new Uint8Array(head.length + Owords.byteLength + lse.byteLength);
      out.set(head, 0);
      out.set(new Uint8Array(Owords.buffer, Owords.byteOffset, Owords.byteLength), head.length);
      out.set(new Uint8Array(lse.buffer, lse.byteOffset, lse.byteLength), head.length + Owords.byteLength);
      send(out);
      break;
    }
    case T.STATS: {
      const s = `state=idle attn_calls=${stats.calls} last_ms=${stats.lastMs.toFixed(3)} held=${heldKeys()} appended=${store.appended} ${kernelStatus()}`;
      send(okMsg(new TextEncoder().encode(s)));
      break;
    }
    case T.PING: {
      let n = 0;
      if (payload.length >= 4) n = dv.getUint32(0, true);
      send(okMsg(new Uint8Array(n).fill(0x5a)));
      break;
    }
    case T.BYE:
      ws.close();
      break;
    default:
      send(errMsg(`unknown message ${type}`));
  }
  refresh();
}

function onData(chunk) {
  const merged = new Uint8Array(rxBuf.length + chunk.length);
  merged.set(rxBuf, 0); merged.set(chunk, rxBuf.length);
  rxBuf = merged;
  // process complete messages in order (strict request -> reply preserved)
  const pump = async () => {
    while (rxBuf.length >= 16) {
      const h = decodeHdr(rxBuf, 0);
      if (h.magic !== MAGIC) { log('bad magic, closing'); ws.close(); return; }
      if (h.len > (1 << 31)) { log('oversize message, closing'); ws.close(); return; }
      if (rxBuf.length < 16 + h.len) return; // wait for more
      const type = h.type;
      const payload = rxBuf.slice(16, 16 + h.len);
      rxBuf = rxBuf.slice(16 + h.len);
      try {
        await handleMessage(type, payload);
      } catch (e) {
        try { send(errMsg(String((e && e.message) || e))); } catch (_) {}
      }
    }
  };
  pump();
}

async function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  const url = $('url').value.trim();
  const token = $('token').value.trim();
  if (!url) { log('enter the bridge ws:// URL first'); return; }
  const full = url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token);
  $('state').textContent = 'connecting…';
  await ensureKernels(log);
  ws = new WebSocket(full);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { $('state').textContent = 'connected'; rxBuf = new Uint8Array(0); log('connected to ' + url); refresh(); };
  // Binary frames are PATN messages; text frames are host notes ("mac PHASE...").
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') { macNote(ev.data); return; }
    onData(new Uint8Array(ev.data));
  };
  ws.onclose = (ev) => {
    $('state').textContent = `closed (${ev.code})`;
    log(`closed (${ev.code}); reconnect in 3 s`);
    setTimeout(() => { if ($('auto').checked) connect(); }, 3000);
  };
  ws.onerror = () => { $('state').textContent = 'error'; };
  refresh();
}

function macNote(line) {
  // :50061-style mac PHASE reports arrive tunneled as TEXT frames "mac ...".
  const m = /^mac (\S+)(.*)$/.exec(line);
  if (m) {
    macPhase = m[1];
    log('mac: ' + line);
    refresh();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  $('connect').addEventListener('click', connect);
  setInterval(refresh, 1000);
  log('enter ws://host:50063 ?token=… and Connect. Needs WebGPU (Chrome/Edge 113+, Safari 26+).');
});
