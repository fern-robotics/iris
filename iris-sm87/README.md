# iris-sm87

SM87-targeted vision-language inference application for Jetson Orin NX.

## Layout

- `src/main.rs`: current LLaVA baseline runner.
- `src/adapters/`: model-family adapters. Each adapter owns its media preprocessing, prompt format, modality-token handling, and multimodal prefill contract.
- `kernels/`: CUDA kernels required by `iris-core` on this focused SM87 build.
- `flash-attn/fa-v1/`: existing Ampere-compatible FlashAttention baseline.
- `flash-attn/fa-sm87/`: reserved for the experimental SM87 forward-attention implementation.
- `flash-attn/fa-v3/`: Hopper-only reference source; excluded from the Cargo workspace and never built for SM87.

`iris-sm87` owns application orchestration. Model adapters and transformer layers select generic attention operations; CUDA kernels remain model-agnostic.
