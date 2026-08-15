# Iris runnable baselines

## Gemma4 CUDA baseline

From the Iris repository root, build on a CUDA machine:

```bash
cargo build --release -p iris-examples --example gemma4 --features cuda
```

Run the text-generation harness:

```bash
cargo run --release -p iris-examples --example gemma4 --features cuda -- \
  --model-id google/gemma-4-E4B-it \
  --prompt 'Explain what CUDA threads are.' \
  --sample-len 64
```

The model files are fetched with `hf-hub` into the standard Hugging Face cache. Ensure you have access to the model repository and set `HF_TOKEN` if required.

This validates the Gemma4 **text-only** decoder path. The first-token top-10 logits for a fixed prompt were checked against Hugging Face Transformers on the RTX 3090, and a chat-formatted CUDA generation run produces coherent text. Image and audio preprocessing/multimodal correctness remain separate work.
