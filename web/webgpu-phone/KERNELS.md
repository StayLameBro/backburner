# Pinned WebGPU kernels (vendored contracts)

Fetched 2026-10-02 from the `webgpu-kernels` org via the same artifact URLs the
`@huggingface/kernels` loader uses:

    https://huggingface.co/kernels/<repo>/resolve/<rev>/build/webgpu/<file>

`version: 1` follows the repo's `v1` branch (moves as fixes land). Below are the
commit revisions those branches pointed at when vendored, so a byte-identical
re-fetch is one `revision:` argument away. All Apache-2.0.

| op | repo | v1 rev (pinned) | manifest | correctness cases |
|---|---|---|---|---|
| MatMul | `webgpu-kernels/ai.onnx.MatMul` | `b2b2761ada6793f50e5c8f17cc1dcbe811c6b95c` | `vendored/ai.onnx.MatMul.manifest.json` | `test.json` on Hub |
| Softmax | `webgpu-kernels/ai.onnx.Softmax` | `1986f903429837098d4e5f0f3ed01481c08c2180` | `vendored/ai.onnx.Softmax.manifest.json` | `test.json` on Hub |
| Add | `webgpu-kernels/ai.onnx.Add` | `abafe9b97aca68b4ca5472cae446238020286db0` | `vendored/ai.onnx.Add.manifest.json` + `vendored/ai.onnx.Add.test.json` | vendored |

Re-fetch exactly:

```bash
REV=b2b2761ada6793f50e5c8f17cc1dcbe811c6b95c  # MatMul v1 as vendored
curl -L "https://huggingface.co/kernels/webgpu-kernels/ai.onnx.MatMul/resolve/$REV/build/webgpu/manifest.json"
```

## Why these three

One attention call over `nk` held keys, per KV head, is:

1. `S = (Q * scale) @ K^T` — `ai.onnx.MatMul`, `[48,256] x [256,nk] -> [48,nk]`
   (48 rows = 8 tokens x GQA 6, head dim 256 — see `phone-attn/phone-attn.h`).
2. `P = softmax(S)` row-wise — `ai.onnx.Softmax`.
3. `O = P @ V` — `ai.onnx.MatMul` again, `[48,nk] x [nk,256] -> [48,256]`.

`lse = max + log(sum(exp))` per row is computed on the CPU from `S`
(cheap next to the matmuls) because the Softmax kernel returns normalized
probabilities only. Key chunks are merged with the same log-sum-exp
`merge_partial` math as `phone-attn.h`, so chunking changes speed, not answers.
`Add` is vendored for the residual/logit merge path and as the broadcast-shape
reference (its manifest + `test.json` document the loader's broadcasting contract).

Manifest excerpts (from the vendored files):

- MatMul: inputs `a`/`b` dtype `T`, output `y` with derived `matmulShape`;
  `T` in `float32, float16, int32, uint32`. We call it float32 throughout v1.
- Softmax: row-wise over the last axis — exactly the `[48,nk]` row layout above.
- Add: multidirectional broadcasting (e.g. `[2,3] + [3] -> [2,3]`), the same
  call pattern as the heavyweight ops.

The browser loads these at runtime with `getKernel("webgpu-kernels/ai.onnx.MatMul",
{ version: 1 })` — no shader toolchain, no server, per the blog. The vendored
copies are the inspectable contract + the pin; they are not executed directly.
