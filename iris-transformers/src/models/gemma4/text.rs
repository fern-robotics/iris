//! Gemma 4 text decoder.
//!
//! and following the candle gemma3.rs patterns.

use std::sync::Arc;

use iris_core::{DType, Device, Module, Result, Tensor, D};
use iris_nn::{linear_b as linear_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

// ── RmsNorm (Gemma-style with +1 offset) ────────────────────────────────────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        // Gemma4 checkpoint RMSNorm weights are direct scales. Unlike older
        // Gemma variants, they must not receive an additional +1 offset.
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

// Gemma4 uses the Llama-style half-split rotation:
// [-x[d/2:], x[:d/2]], not Iris's generic interleaved RoPE helper.
fn apply_gemma4_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;
    let rotate_half = |x: &Tensor| -> Result<Tensor> {
        let dim = x.dim(D::Minus1)?;
        let x1 = x.narrow(D::Minus1, 0, dim / 2)?;
        let x2 = x.narrow(D::Minus1, dim / 2, dim - dim / 2)?;
        Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
    };
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

// ── RotaryEmbedding (standard, for sliding layers) ──────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        apply_gemma4_rope(q, k, &cos, &sin)
    }
}

// ── ProportionalRotaryEmbedding (for global/full layers) ────────────────────

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        apply_gemma4_rope(q, k, &cos, &sin)
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("gate_proj"))?;
        let up_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("up_proj"))?;
        let down_proj = linear_bias(intermediate_size, hidden_size, bias, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ── Flash attention ─────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    iris_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

// ── KvCache ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvCache {
    Normal(iris_nn::kv_cache::KvCache),
    Rotating(iris_nn::kv_cache::RotatingKvCache),
}

// ── Attention ───────────────────────────────────────────────────────────────

type SharedKv = (Tensor, Tensor);

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Option<Linear>,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    is_kv_shared_layer: bool,
    store_full_length_kv: bool,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    use_flash_attn: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let is_sliding = cfg.is_sliding(layer_idx);

        let (head_dim, num_kv_heads) = if is_sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = cfg
                .num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads);
            (cfg.global_head_dim, global_kv)
        };

        let num_kv_groups = num_heads / num_kv_heads;
        let first_kv_shared_layer = cfg.num_hidden_layers.saturating_sub(cfg.num_kv_shared_layers);
        let is_kv_shared_layer = layer_idx >= first_kv_shared_layer && cfg.num_kv_shared_layers > 0;
        let store_full_length_kv = !is_kv_shared_layer
            && cfg.layer_types[..first_kv_shared_layer]
                .iter()
                .rposition(|layer_type| layer_type == &cfg.layer_types[layer_idx])
                == Some(layer_idx);
        let q_proj = linear_bias(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let (k_proj, v_proj, k_norm) = if is_kv_shared_layer {
            (None, None, None)
        } else {
            (
                Some(linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?),
                Some(linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?),
                Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?),
            )
        };
        let o_proj = linear_bias(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;

        let kv_cache = if is_sliding {
            KvCache::Rotating(iris_nn::kv_cache::RotatingKvCache::new(
                2,
                cfg.effective_sliding_window(),
            ))
        } else {
            KvCache::Normal(iris_nn::kv_cache::KvCache::new(
                2,
                cfg.max_position_embeddings,
            ))
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            is_sliding,
            is_kv_shared_layer,
            store_full_length_kv,
            rotary_emb_global,
            rotary_emb_local,
            kv_cache,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        shared_kv: Option<&SharedKv>,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Option<SharedKv>)> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self
            .q_norm
            .forward(&self.q_proj.forward(xs)?.reshape((b_sz, q_len, self.num_heads, self.head_dim))?.transpose(1, 2)?)?;

        let (k, v, stored_shared_kv) = if self.is_kv_shared_layer {
            let (k, v) = shared_kv.ok_or_else(|| {
                iris_core::Error::msg("Gemma4 shared-KV layer ran before its source K/V state")
            })?;
            (k.clone(), v.clone(), None)
        } else {
            let k_proj = self.k_proj.as_ref().expect("non-shared layer has K projection");
            let v_proj = self.v_proj.as_ref().expect("non-shared layer has V projection");
            let k_norm = self.k_norm.as_ref().expect("non-shared layer has K norm");
            let k = k_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k_norm.forward(&k)?;
            let v = v_norm(&v, self.rms_norm_eps)?;
            let (_, k) = if self.is_sliding {
                // Query is already rotated below; only rotate this newly computed K.
                self.rotary_emb_local.apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
            } else {
                self.rotary_emb_global.apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
            };
            let (k, v) = match &mut self.kv_cache {
                KvCache::Normal(cache) => cache.append(&k, &v)?,
                KvCache::Rotating(cache) => cache.append(&k, &v)?,
            };
            let stored = self.store_full_length_kv.then(|| (k.clone(), v.clone()));
            (k, v, stored)
        };

        let q = if self.is_sliding {
            self.rotary_emb_local.apply_rotary_emb_qkv(&q, &q, seqlen_offset)?.0
        } else {
            self.rotary_emb_global.apply_rotary_emb_qkv(&q, &q, seqlen_offset)?.0
        };
        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let attn_output = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            flash_attn(&q, &k, &v, 1.0, mask.is_some())?.transpose(1, 2)?
        } else {
            // Gemma4 normalizes Q and K explicitly, so eager attention uses
            // unit scaling rather than the conventional 1/sqrt(head_dim).
            let attn_weights = q.matmul(&k.transpose(2, 3)?)?;

            let attn_weights = match mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = iris_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        let output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)?;
        Ok((output, stored_shared_kv))
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    // Gemma4's per-layer embeddings add a token- and context-dependent
    // residual after the MLP block.
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RmsNorm>,
    per_layer_activation: Option<Activation>,
    layer_scalar: Tensor,
    #[allow(dead_code)]
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding(layer_idx);
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            vb.pp("self_attn"),
        )?;
        let mlp = MLP::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_activation,
            false,
            vb.pp("mlp"),
        )?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm, per_layer_activation) =
            if cfg.hidden_size_per_layer_input > 0 {
                let ple_dim = cfg.hidden_size_per_layer_input;
                (
                    Some(iris_nn::linear_no_bias(
                        cfg.hidden_size,
                        ple_dim,
                        vb.pp("per_layer_input_gate"),
                    )?),
                    Some(iris_nn::linear_no_bias(
                        ple_dim,
                        cfg.hidden_size,
                        vb.pp("per_layer_projection"),
                    )?),
                    Some(RmsNorm::new(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb.pp("post_per_layer_input_norm"),
                    )?),
                    Some(cfg.hidden_activation),
                )
            } else {
                (None, None, None, None)
            };
        let layer_scalar = vb.get(1, "layer_scalar")?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            per_layer_activation,
            layer_scalar,
            is_sliding,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        shared_kv: Option<&SharedKv>,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Option<SharedKv>)> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, stored_shared_kv) = self.self_attn.forward(
            &xs,
            shared_kv,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let xs = (residual + xs)?;

        let xs = match (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            &self.per_layer_activation,
            per_layer_input,
        ) {
            (Some(gate), Some(projection), Some(norm), Some(activation), Some(per_layer_input)) => {
                let residual = &xs;
                let ple = gate.forward(&xs)?.apply(activation)?;
                let ple = (ple * per_layer_input)?;
                let ple = projection.forward(&ple)?.apply(norm)?;
                (residual + ple)?
            }
            _ => xs,
        };
        Ok((xs.broadcast_mul(&self.layer_scalar)?, stored_shared_kv))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────────

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

// ── TextModel ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: iris_nn::Embedding,
    // Packed [vocab, num_layers * per_layer_dim] auxiliary embedding table.
    per_layer_embed_tokens: Option<iris_nn::Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<RmsNorm>,
    per_layer_input_dim: usize,
    layer_is_sliding: Vec<bool>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
}

impl TextModel {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        // `vb` is the language-model root. Callers pass either
        // `model.language_model` (full multimodal checkpoint) or an equivalent
        // text-only root, so do not add another `model` path component here.
        let vb_m = vb;
        let embed_tokens =
            iris_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let (per_layer_embed_tokens, per_layer_model_projection, per_layer_projection_norm) =
            if cfg.hidden_size_per_layer_input > 0 {
                let packed_dim = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
                (
                    Some(iris_nn::embedding(
                        cfg.vocab_size_per_layer_input,
                        packed_dim,
                        vb_m.pp("embed_tokens_per_layer"),
                    )?),
                    Some(iris_nn::linear_no_bias(
                        cfg.hidden_size,
                        packed_dim,
                        vb_m.pp("per_layer_model_projection"),
                    )?),
                    Some(RmsNorm::new(
                        cfg.hidden_size_per_layer_input,
                        cfg.rms_norm_eps,
                        vb_m.pp("per_layer_projection_norm"),
                    )?),
                )
            } else {
                (None, None, None)
            };

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb_m.dtype(),
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);
        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb_m.dtype(),
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb_global.clone(),
                rotary_emb_local.clone(),
                cfg,
                layer_idx,
                vb_l.pp(layer_idx),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            iris_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_m.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            per_layer_embed_tokens,
            per_layer_model_projection,
            per_layer_projection_norm,
            per_layer_input_dim: cfg.hidden_size_per_layer_input,
            layer_is_sliding: (0..cfg.num_hidden_layers).map(|idx| cfg.is_sliding(idx)).collect(),
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb_m.device().clone(),
            dtype: vb_m.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }
        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;
        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        Ok((Some(mask), Some(sliding_mask)))
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        xs * (self.hidden_size as f64).sqrt()
    }

    fn token_per_layer_inputs(&self, input_ids: &Tensor) -> Result<Option<Tensor>> {
        let Some(embedding) = &self.per_layer_embed_tokens else {
            return Ok(None);
        };
        let (batch_size, seq_len) = input_ids.dims2()?;
        let inputs = (embedding.forward(input_ids)? * (self.per_layer_input_dim as f64).sqrt())?;
        inputs.reshape((batch_size, seq_len, self.layers.len(), self.per_layer_input_dim)).map(Some)
    }

    fn project_per_layer_inputs(
        &self,
        input_embeds: &Tensor,
        token_inputs: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        let (Some(projection), Some(norm)) = (
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) else {
            return Ok(None);
        };
        let (batch_size, seq_len, _) = input_embeds.dims3()?;
        let projected = (projection.forward(input_embeds)? * (self.hidden_size as f64).sqrt().recip())?;
        let projected = projected.reshape((
            batch_size,
            seq_len,
            self.layers.len(),
            self.per_layer_input_dim,
        ))?;
        let projected = norm.forward(&projected)?;
        match token_inputs {
            Some(token_inputs) => ((projected + token_inputs)? * 2f64.sqrt().recip()).map(Some),
            None => Ok(Some(projected)),
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens(input_ids)?;
        let token_inputs = self.token_per_layer_inputs(input_ids)?;
        self.forward_embeds_with_per_layer_inputs(
            &xs,
            token_inputs.as_ref(),
            seqlen_offset,
            batch_size,
            seq_len,
        )
    }

    pub fn forward_embeds(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Multimodal callers supply modified embeddings. They have no exact token
        // identity for media positions, so use the context projection component.
        self.forward_embeds_with_per_layer_inputs(xs, None, seqlen_offset, batch_size, seq_len)
    }

    fn forward_embeds_with_per_layer_inputs(
        &mut self,
        xs: &Tensor,
        token_inputs: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;
        let per_layer_inputs = self.project_per_layer_inputs(xs, token_inputs)?;

        let mut xs = xs.clone();
        let mut shared_sliding_kv: Option<SharedKv> = None;
        let mut shared_full_kv: Option<SharedKv> = None;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let per_layer_input = match &per_layer_inputs {
                Some(inputs) => Some(inputs.narrow(2, layer_idx, 1)?.squeeze(2)?),
                None => None,
            };
            let shared_kv = if self.layer_is_sliding[layer_idx] {
                shared_sliding_kv.as_ref()
            } else {
                shared_full_kv.as_ref()
            };
            let (next_xs, stored_shared_kv) = layer.forward(
                &xs,
                per_layer_input.as_ref(),
                shared_kv,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
            )?;
            xs = next_xs;
            if let Some(shared_kv) = stored_shared_kv {
                if self.layer_is_sliding[layer_idx] {
                    shared_sliding_kv = Some(shared_kv);
                } else {
                    shared_full_kv = Some(shared_kv);
                }
            }
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
