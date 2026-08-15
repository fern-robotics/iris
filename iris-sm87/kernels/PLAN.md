# SM87 FlashAttention Plan

## Goal

Build a forward-only, inference-oriented FlashAttention-style CUDA kernel for the Jetson Orin NX GPU (SM87), then use it in real VLM prefill workloads.

This is **not** a literal FlashAttention-3 or FlashAttention-4 port. FA3 needs Hopper hardware; FA4 needs Blackwell hardware. The implementation target is an Ampere/SM87 kernel informed primarily by FlashAttention-2, with carefully measured ideas from FA4.

## Hardware contract

SM87 provides FP16/BF16 tensor cores, shared memory, normal CUDA warps, and `cp.async` global-to-shared copies. It does not provide TMA, WGMMA/warpgroup MMA, TMEM, Blackwell `tcgen05`, CTA clusters, or 2-CTA MMA.

## Reading order

1. `../../.references/OnlineSoftmax-1805.02867.pdf`
2. `../../.references/FlashAttention-2205.14135.pdf`
3. `../../.references/FlashAttention2-2307.08691.pdf`
4. `../../.references/FlashAttention3-2407.08608.pdf` for non-portable Hopper design context.
5. `../../.references/FlashAttention4-2603.05451v1.pdf` sections on roofline analysis and conditional rescaling only.
6. `../../.references/NVIDIA-CUDA-C-Programming-Guide.pdf`
7. `../../.references/Ampere-Microbenchmarking-2208.11174.pdf`

Read the local FA-v1 source as a structural reference, not as a first implementation target:

- `../flash-attn/fa-v1/kernels/flash_api.cu`
- `../flash-attn/fa-v1/kernels/flash_fwd_launch_template.h`
- `../flash-attn/fa-v1/kernels/flash_fwd_kernel.h`
- one `flash_fwd_hdim128_fp16*_sm80.cu` instantiation

## Baseline workload

Use LLaVA decoder prefill as the first real workload:

- FP16
- batch size 1
- causal self-attention
- `[B, S, H, D]` at the attention-kernel interface
- head dimension 128
- start with equal Q and KV head counts
- image-token sequence plus prompt tokens

Benchmark vision attention, language prefill, and language decode separately. The first kernel targets language prefill. Decode (`Q_len = 1`) is a separate KV-cache bandwidth problem.

## Phases

### 0. Establish a reproducible baseline

- Run the VLM with ordinary SDPA.
- Record model, dtype, image resolution, image-token count, prompt length, generated-token count, memory use, prefill latency, and decode tokens/s.
- Use CUDA events for timings, with warm-up and no allocations inside the measured loop.
- Capture an Nsight Systems timeline before optimizing.

**Exit criterion:** a saved baseline report for one LLaVA prompt/image case.

### 1. Learn the primitives independently

Implement and validate these exercises outside the VLM path:

1. coalesced vector/matrix loads;
2. shared-memory tiled FP32 GEMM;
3. stable row-wise softmax;
4. tiled online softmax;
5. fused attention forward without materializing the score matrix.

Use a CPU/ordinary-SDPA result as the oracle at every stage.

**Exit criterion:** explain the online-softmax recurrence and validate it on tiled random inputs.

### 2. Minimal attention kernel

Implement exactly one forward kernel variant:

- F16 input/output;
- FP32 score, maximum, normalizer, and output accumulation;
- `D = 128`;
- contiguous Q/K/V;
- batch 1;
- no dropout, ALiBi, paging, variable lengths, or GQA;
- causal and non-causal modes.

One CTA owns `(batch, head, query_tile)`. It loops over K/V tiles and maintains, per query row:

```text
m: running score maximum
l: running sum of exp(score - m)
o: running weighted-V accumulator
```

For each K/V tile, update the online-softmax state instead of storing the score or probability matrix. Apply the causal mask before the row maximum and exponentiation.

**Exit criterion:** matches the SDPA reference for sequence lengths crossing tile boundaries: 63, 64, 65, 127, 128, and 129.

### 3. Correctness matrix

Add deterministic and random tests for:

- non-causal prefill;
- causal prefill;
- causal decode with `Q_len = 1` and `KV_len > 1`;
- odd/non-tile-aligned sequence lengths;
- multiple heads;
- F16 numerical tolerance against an FP32 reference.

Do not add a performance optimization without preserving this matrix.

**Exit criterion:** all supported shapes pass reproducibly.

### 4. SM87 optimization sequence

Change one item at a time and retain only measured improvements:

1. coalesced global memory accesses;
2. shared-memory K/V tiling;
3. tune `BLOCK_M` and `BLOCK_N`;
4. double-buffer K/V tiles;
5. use `cp.async` for global-to-shared staging;
6. use FP16 tensor-core MMA (`mma.sync` or an appropriate CUTLASS/CuTe Ampere path);
7. tune register pressure, occupancy, and shared-memory use.

Use Nsight Compute to check tensor-core utilization, achieved memory bandwidth, occupancy, register spills, shared-memory bank conflicts, and warp stalls.

**Exit criterion:** improvement over the ordinary SDPA baseline for target prefill shapes without a correctness regression.

### 5. Carefully evaluate FA4-derived ideas

Evaluate independently after the conventional online-softmax kernel is correct:

- conditional online-softmax rescaling;
- alternative work-tile scheduling;
- limited software exponential experiments only if profiling shows exponentiation is a bottleneck.

Do not attempt TMEM, asynchronous Blackwell MMA, 2-CTA MMA, TMA, or Blackwell tile instructions on SM87.

**Exit criterion:** keep an idea only with a measured end-to-end prefill improvement on Orin.

### 6. Expand supported attention contracts

Add features in this order only when a target VLM requires them:

1. direct GQA without materializing repeated K/V heads;
2. head dimensions 64 and 256;
3. sliding-window causal masking for Gemma4 local layers;
4. cache-offset/prefill variants;
5. a dedicated decode kernel;
6. variable-length batch support;
7. paged KV cache, if serving requires it.

Each new contract needs an SDPA-reference test and benchmark case.

### 7. Integrate with VLM inference

The kernel remains model-agnostic. Model adapters own media preprocessing, token formatting, and embedding injection. Transformer attention selects the shared SM87 backend only when its dtype, layout, mask, head dimension, and cache contract are supported; otherwise it falls back to ordinary attention.

Initial integration order:

1. LLaVA language-decoder prefill;
2. LLaVA vision encoder attention;
3. Gemma4 global decoder layers;
4. Gemma4 sliding-window layers;
5. later Qwen-family adapters.

**Exit criterion:** LLaVA produces equivalent generated output with the SM87 attention feature enabled and has a documented prefill benchmark improvement.

## Non-goals for the first implementation

- backward/training kernels;
- FA3 Hopper code porting;
- FA4 Blackwell code porting;
- FP8/FP4 paths;
- dropout, ALiBi, paged KV, varlen batching, and GQA before the minimal kernel is correct;
- optimizing `Q_len = 1` decode before prefill is measured.

## Per-change checklist

Before accepting a kernel change, record:

- exact Q/K/V shape, dtype, mask, and cache offset;
- max/mean numerical error against reference;
- warm-up count and timed iteration count;
- CUDA-event latency;
- memory allocation behavior;
- Nsight profile observations;
- isolated attention result and end-to-end VLM result.
