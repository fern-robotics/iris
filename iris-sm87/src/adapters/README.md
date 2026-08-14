# Model adapters

Add one adapter per VLM family here. An adapter converts user media and a prompt into the model-specific prefill input, then hands cached decoding back to the shared generation runtime.

Keep model-specific token formats, media preprocessing, modality embedding injection, and cache-reset rules here. Do not put SM87 CUDA kernel code in adapters.
