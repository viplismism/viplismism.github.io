---
layout: post
title: "FP8 KV cache quantization: how a bad scale breaks everything"
description: how FP8 KV cache quantization in vLLM can go wrong when scales are locked, defaulted, or computed from bad activations - from quiet clipping to NaN propagation to output that looks like "!!!!!!!!!!!!!!!!".
tags: [vLLM, Quantization, Inference, Deep Learning]
version: Draft
release: 17-03-2026
---

so at the end of my [attention post](https://viplismism.github.io/2026/02/22/attention), i mentioned kv-cache and inference optimizations. this is that follow-up - except it turned into something i didn't expect. what started as "let me explain kv-cache quantization" became a real investigation into a production bug where a multimodal model just... starts outputting `"!!!!!!!!!!!!!!!!!!!"`. yeah. literally exclamation marks. nothing else.

i've written a [separate post on FP8 fundamentals](/2026/03/15/fp8-how-numbers-live-in-8-bits) that covers the bit format, exponent bias, subnormals, all of that. if you haven't read it, that's a good starting point. this post assumes you know what FP8 E4M3 is and picks up from there: what happens when you use it to compress a KV cache and something goes wrong.

the shape of the bug is simple, but the details are annoying: FP8 is not the villain by itself. the scale is. if the scale is too small, values clip. if the scale is too large, you waste most of the tiny FP8 grid. and if the scale itself becomes NaN or inf, the model is basically cooked until the server state is reset.

there's also a tokenizer detour near the end explaining why the garbage output is specifically `!` and not some other character. it's not required for understanding the quantization bug, but honestly i find it interesting enough to keep in here.

## quick FP8 recap

FP8 E4M3 is an 8-bit floating point format: 1 sign bit, 4 exponent bits, 3 mantissa bits. the max representable value is 448 (that's 2⁸ × 1.75 - the full derivation is in [the other post](/2026/03/15/fp8-how-numbers-live-in-8-bits)). the NaN pattern is when the exponent AND mantissa are all ones (`1111 111`), for either sign bit. everything else is a valid number, no infinity.

the thing that matters for this post: FP8 has a very narrow range. max 448. if your values are larger than that, they clip. and because the mantissa is only 3 bits, the gap between adjacent representable values near 448 is 32 - meaning 416 and 448 are neighbors with nothing in between.

one small note: vLLM can also use FP8 E5M2 in some paths. E5M2 has a much wider range and less precision. the exact max value changes, but the lesson doesn't: an FP8 value is only meaningful together with the scale used to encode and decode it.

this is fine when values stay predictable. it's not fine when they don't. that's the whole post.

## the vLLM KV cache bug

### the problem

when running Qwen-family multimodal models with FP8 KV cache quantization in vLLM, people have reported two flavors of failure:

1. **finite garbage** - wrong colors, wrong objects, repeated characters, weird but still finite output.
2. **NaN garbage** - the model collapses into repeated punctuation like `"!!!!!!!!!!!!!!!!!!!!!!"`.

these look similar from the outside because both are "the model is broken now." internally they're different. finite garbage usually means your values got badly rounded or clipped. `!!!!` strongly suggests NaNs reached the logits or the sampler hit a NaN edge case.

the configuration i care about here is the dynamic-scale path:

```yaml
args: [
  '--kv-cache-dtype', 'fp8',
  '--calculate-kv-scales',
  # ... other args
]
```

two flags. that's all it took to enter the risky path. let me show you why.

### vLLM's scale calculation: the risky part

you can't just shove FP16 values directly into FP8 and call it a day. you need a **scale factor** to map your actual values into FP8's representable range. for E4M3, that range tops out at 448. vLLM can compute these scales dynamically during the first forward pass. every time `forward()` runs on an attention layer, it checks whether scales need computing:

```python
# from vllm/model_executor/layers/attention/attention.py

class Attention(nn.Module, AttentionLayerBase):
    def forward(self, query, key, value, ...):
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(
                query, key, value, self.layer_name
            )
        # ... rest of attention computation
```

that `maybe_calc_kv_scales` is a gate - once a layer has calculated its scales, it stops recalculating them:

```python
def maybe_calc_kv_scales(query, key, value, layer_name):
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    # Only calculate if the layer's flag is True
    # This flag gets set to False after the first forward pass
    if not self.calculate_kv_scales:
        return

    self.calc_kv_scales(query, key, value)
```

and here's the actual scale computation:

```python
def calc_kv_scales(self, query, key, value):
    self._q_scale.copy_(torch.abs(query).max() / self.q_range)
    self._k_scale.copy_(torch.abs(key).max() / self.k_range)
    self._v_scale.copy_(torch.abs(value).max() / self.v_range)
    self._q_scale_float = self._q_scale.item()
    self._k_scale_float = self._k_scale.item()
    self._v_scale_float = self._v_scale.item()
    # We only calculate the scales once
    self.calculate_kv_scales = False  # ← LOCKED FOREVER
```

see that last line? `self.calculate_kv_scales = False`. once a layer has calculated its scales, that layer keeps them. the divisors are environment constants:

```python
# from vllm/envs.py
Q_SCALE_CONSTANT: int = 200
K_SCALE_CONSTANT: int = 200
V_SCALE_CONSTANT: int = 100
```

so the scale formula is roughly: `k_scale = max(abs(key_values)) / 200`. if the max absolute key value in the first calibration pass is 5.0, then `k_scale = 5.0 / 200 = 0.025`.

and the dequantization uses this scale on every single cache read:

```python
# from vllm/v1/attention/ops/chunked_prefill_paged_decode.py

K_load = tl.load(key_cache_ptr + k_offset, ...)

if K_load.dtype.is_fp8():
    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
```

load the FP8 value, cast to float32, multiply by the scale. every read. so if the scale is stale, too small, too large, zero, inf, or NaN, every future KV-cache read inherits that mistake. the FP8 byte by itself is not the value. the byte plus the scale is the value.

### why one-time scale calibration is dangerous

imagine this scenario:

```
request 1: K values range [-5.0, +5.0]
  k_scale = 5.0 / 200 = 0.025
  max storable = 448 × 0.025 = 11.2
  ✓ works fine, all values fit

request 2: K values range [-20.0, +20.0]
  k_scale STILL = 0.025 (locked!)
  value 20.0 → 20.0 / 0.025 = 800
  but FP8 max = 448
  CLIPPED to 448!
  retrieved: 448 × 0.025 = 11.2

  ALL values > 11.2 become exactly 11.2!
```

and it works the other way too:

```
request 1: K values range [-50.0, +50.0]
  k_scale = 50.0 / 200 = 0.25
  max storable = 448 × 0.25 = 112
  ✓ no clipping

request 2: K values range [-2.0, +2.0]
  k_scale STILL = 0.25 (locked!)
  value 2.0 → 2.0 / 0.25 = 8
  only uses 8 out of 448 FP8 levels!
  severe precision loss
```

the first scenario is clipping. the second is precision loss. both are bad, but they fail differently. clipping gives you loud, visible corruption. precision loss usually gives you softer quality degradation. neither one requires NaN yet.

### the clipping: where things start going wrong

the clipping happens at the hardware level. here's the actual CUDA code that does the FP8 conversion:

```cpp
// from vllm/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh

template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}
```

`__NV_SATFINITE` is NVIDIA's saturation mode - it clips values exceeding the FP8 range to the largest finite FP8 value instead of producing infinity. for E4M3, that ceiling is 448. it's a safety mechanism, but "safe" here only means "finite." it does not mean "semantically useful."

here's what the error looks like at different value levels:

| original value | scaled (÷0.025) | after clipping | retrieved (×0.025) | error |
|----------------|-----------------|----------------|-------------------|-------|
| 5.0 | 200 | 200 | 5.0 | 0% |
| 11.2 | 448 | 448 | 11.2 | 0% |
| 15.0 | 600 | 448 | 11.2 | 25.3% |
| 20.0 | 800 | 448 | 11.2 | 44.0% |
| 50.0 | 2000 | 448 | 11.2 | 77.6% |

anything above 11.2 gets clamped to exactly 11.2. now imagine an image with rich, diverse feature values where a bunch of them are above 11.2. they ALL become the same number.

### the identical values problem

when multiple tokens (especially image tokens in multimodal models) have K values above the clipping threshold, they ALL collapse to the same value:

```
original image token K values (diverse):
  [15.3, 18.7, 12.1, 19.5, 16.8, 20.0, 14.2, 17.9]

after FP8 quantization + clipping:
  [11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2]
```

all diversity gone. all information gone. just a flat line of identical values.

now here's the thing - identical K values don't directly crash the model. softmax handles finite inputs fine. what you get instead is kinda the worst outcome: the attention mechanism **can't distinguish those tokens well anymore**. if a bunch of keys collapse toward the same ceiling, their dot products become much less informative, so attention starts averaging information that used to be separate. the model can still sound coherent, but it's now coherent in the wrong direction - wrong colors, wrong objects, repeated characters.

that's the finite-corruption story, and it's the kind of thing reported in vLLM issue #41343. important caveat: that issue is about an `fp8_e5m2` KV-cache path with default scales, not this exact E4M3 dynamic-scale path. i still think it's useful evidence because it shows the same general failure class: bad FP8 scaling can silently turn vision features into bad attention.

but `!!!!` is a different smell. `!!!!` points much more strongly at NaN.

### so where does the NaN actually come from?

okay, this is the part where i have to be honest - because this is where the original version of this post got it wrong, and i'd rather just say that than paper over it.

the clipping itself doesn't make a NaN. clipping shoves everything to 448, and 448 is a perfectly normal finite number. identical finite K values produce wrong attention outputs, but softmax still runs fine on them - you get garbage attention weights, but finite garbage. no NaN.

the tell that NaN is involved is the collapse pattern. finite corruption can make the model wrong, repetitive, or weird, but the math is still finite. NaN is different. once a NaN reaches attention scores or logits, normal arithmetic stops behaving like normal arithmetic.

NaN in the scale or KV cache poisons every query that touches it. `anything × NaN = NaN`. doesn't matter how clean the new input is.

so where does the NaN enter? the prime suspect is the **scale** itself:

```python
k_scale = abs(key).max() / 200
```

this runs once for the layer. if that calibration pass sees unusual values - activations from an uninitialized recurrent state, a particular input that drives an intermediate value toward inf, or a degenerate case that produces a zero/invalid scale - the scale can come out as NaN, inf, or zero. and from that point on, every dequantized K on that path can become:

```
fp8_value × NaN = NaN
```

vLLM issue #37554 is the closest version of this story: garbage from an uninitialized GDN recurrent state in Qwen3.5 interacted badly with `--calculate-kv-scales`, and the practical fix was to remove that flag. the scale path was the danger zone.

i'll be straight: i did not catch the exact instruction where the first NaN is born in a debugger. so treat "NaN enters through the scale" as the strongest supported explanation, not as a profiler screenshot. what i *am* sure about is everything after a NaN exists - and that part is brutal.

### NaN propagation: one bad apple spoils the batch

a single NaN key in the cache corrupts everything in one layer. here's the chain.

**starting point:** say token 2 (an image token) has NaN K values in the cache. the other tokens are fine.

**attention scores (QK^T):** every query dot-products with every key:

```
        k₀    k₁    k₂    k₃
                    ↑
                (all NaN)
    ┌────────────────────────────
q₀  │  s₀₀   s₀₁   NaN   s₀₃  │
q₁  │  s₁₀   s₁₁   NaN   s₁₃  │
q₂  │  s₂₀   s₂₁   NaN   s₂₃  │
q₃  │  s₃₀   s₃₁   NaN   s₃₃  │
    └────────────────────────────
```

column 2 is NaN because k₂ is NaN. every single row now has at least one NaN.

**softmax: the point of no return**

```
softmax(x)_i = exp(x_i) / Σⱼ exp(x_j)
```

for row 0: [s₀₀, s₀₁, NaN, s₀₃]

```
denominator = exp(s₀₀) + exp(s₀₁) + NaN + exp(s₀₃)
            = (some value) + NaN
            = NaN

softmax = [NaN, NaN, NaN, NaN]
```

every row has a NaN in column 2. every row's softmax denominator is NaN. every attention weight becomes NaN. one corrupted token just made ALL attention weights NaN.

**attention output → residual:** NaN weights × values = NaN for all tokens. even the ones that had perfectly normal inputs:

```
token 0: [0.5, -0.3, 0.8, ...] + [NaN, NaN, NaN, ...]
       = [NaN, NaN, NaN, ...]  ← normal + NaN = NaN
```

```
start of layer:     1 token's keys are NaN
after QK^T:         NaN column in attention scores
after softmax:      ALL rows become NaN (100%)
after attention:    ALL tokens' outputs are NaN
after residual:     100% NaN - corruption complete
```

in ONE layer. from one bad key.

### why future requests can fail too

there are two ways the corruption can outlive the request that triggered it.

first, the scale is layer state. if the locked scale becomes NaN, inf, or just badly wrong, new requests keep using that bad scale. even a perfectly normal text request can produce poisoned KV reads because the decode path still multiplies by the bad scale:

```
new request arrives (perfectly valid text):
1. FP8 cache value × bad locked scale = NaN or badly wrong value
2. attention scores become NaN or distorted
3. output becomes NaN garbage or finite garbage

result: "!!!!!!!!!!!!!!!!!!!"
```

that is why restart helps: it clears the bad layer state and recomputes from a clean process.

second, if **prefix caching** is enabled, the actual KV entries can persist too. vLLM can cache KV entries for common prompt prefixes so they don't need recomputing. if NaN-corrupted entries get saved as a prefix cache entry, any future request with a matching prefix reuses the poisoned entries directly. the corruption gets baked in.

---

**sidebar: why the garbage token is specifically `!`**

this part isn't required reading for the bug, but it's interesting - two independent design decisions from the 1960s and early 2000s line up to produce `!` specifically.

**NaN can break argmax/sampling.** by this point, every logit (the model's score for each possible next token) may be NaN. to pick the next token, some code paths effectively compare scores to find the highest one. but NaN has a weird property - ordinary comparisons with NaN return false.

```
NaN > 5.0?      false
NaN > -1000.0?  false
NaN > NaN?      also false
```

in a naive argmax implementation, that means nothing ever "wins." the current best never updates. argmax starts at index 0, and index 0 is what it returns. real inference stacks have different kernels and samplers, so don't read this as a universal law. read it as the common shape of this failure: NaN logits can collapse token selection into a low-index default.

**token 0 is `!` in Qwen-style byte vocabularies.** this comes from how the tokenizer was built. byte-level BPE tokenizers start with base tokens for byte values, because UTF-8 encodes text as bytes between 0 and 255. if you have a token for every byte value, you can tokenize arbitrary input. nothing is ever truly "unknown."

some of those 256 bytes are invisible control characters (null, tab, newline, space), so GPT-2 remapped them to obscure unicode stand-ins for display purposes. then the remaining bytes get ordered starting from the first printable ASCII character.

the first printable ASCII character is byte 33: `!`.

```python
# from the tokenizer source (bytes_to_unicode function)
bs = list(range(ord("!"), ord("~") + 1))  # starts at "!" (33)

# token 0 → "!"  (byte 33)
# token 1 → '"'  (byte 34)
# token 2 → "#"  (byte 35)
# ...
# token 256 onwards → BPE merges (common word combinations)
```

so `!` isn't special or meaningful - it's just the first printable ASCII character, which makes it the first entry in this style of base vocabulary. if the sampler falls back to token ID 0, token 0 decodes as `!`. you get `!!!!` forever.

(BPE merges - common sequences like "the", "ing", "hello" that get combined into single tokens - get added after all 256 base bytes, so they start at token ID 256 and up. the base byte ordering is what matters here.)

if this were a model with `<pad>` or `<unk>` at index 0, you'd get that instead. with GPT-2/Qwen-style byte-level BPE, the low-index default can visibly show up as `!`.

---

### the complete causal chain

```
1. --kv-cache-dtype fp8 + --calculate-kv-scales enabled
       ↓
2. calibration pass computes K/V scales and stores them on the layer
       ↓
3. either:
   A. scale is finite but wrong for later values
   B. scale itself becomes NaN/inf/zero from bad activations
       ↓
   [A: values clip or waste FP8 levels → finite but wrong attention]
   [B: NaN/inf scale path →]
4. dequantized K/V values become NaN
       ↓
5. NaN column in QK^T attention scores
       ↓
6. softmax denominator = NaN → ALL attention weights become NaN
       ↓
7. ALL tokens' outputs become NaN after attention + residual
       ↓
8. bad scale state or cached prefix entries persist
       ↓
9. future requests reuse the bad state → corruption continues
       ↓
10. NaN logits → sampler collapses to token 0 → token 0 is "!" → "!!!!!!!!!!!!!!!!!!!"
```

10 steps from a config flag to garbage. annoying, but not mystical.

### the fix

```yaml
# instead of:
args: [
  '--kv-cache-dtype', 'fp8',
  '--calculate-kv-scales',
]

# use:
args: [
  '--kv-cache-dtype', 'auto',
  # remove --calculate-kv-scales entirely
]
```

`--kv-cache-dtype auto` stores the KV cache in its original precision (usually FP16 or BF16). no FP8 KV scale, no FP8 clipping, no FP8-induced NaN path. yeah you use more memory (roughly 2× for the cache compared with FP8), but the model actually works. for multimodal models where image tokens and text tokens have very different value distributions, this is the conservative option.

could vLLM fix FP8 KV caching properly? totally. a few ideas:
- **recalculate scales per request** - update them as new value ranges come in instead of locking
- **per-token dynamic quantization** - each token gets its own scale instead of one global one
- **outlier-aware quantization** - detect values that would clip and handle them separately

for now though, the practical debugging rule is simple: if a multimodal model starts producing punctuation soup after enabling FP8 KV cache, remove dynamic KV-scale calculation first. then test FP8 again only after you understand which scale path you're actually using.

**tracked issues:** vLLM [#41343](https://github.com/vllm-project/vllm/issues/41343) (Qwen2-VL/Qwen2.5-VL, `fp8_e5m2` KV cache, finite garbage output - wrong colors, repeated characters), vLLM [#37554](https://github.com/vllm-project/vllm/issues/37554) (Qwen3.5, garbage output around uninitialized GDN recurrent state and `--calculate-kv-scales`, practical fix: remove `--calculate-kv-scales`).

## takeaways

1. **the scale is everything** - it's the ruler you use to fit real values into FP8's tiny grid. set it wrong, default it wrong, or compute it from bad activations, and everything downstream inherits the mistake.

2. **clipping is quiet** - `__NV_SATFINITE` clips to 448 silently. no error, no warning. you get wrong output and have no clue why.

3. **identical clipped values break attention** - when many tokens' K values collapse to the same ceiling, attention can't distinguish them. it weights them equally and averages their contribution. the output is coherent, confident, and wrong.

4. **NaN is more infectious than it looks** - one NaN key in one cached position → NaN column in attention scores → NaN in every softmax denominator → 100% NaN attention weights → 100% NaN output. all in one layer.

5. **server state matters** - locked scales and prefix cache entries can outlive the request that created them. normally that's a feature. when corruption gets in, it's a bug amplifier.

so yeah - two command line flags, a scale that becomes the ruler for the cache, and the fundamental limitations of 8-bit floating point combine to produce `"!!!!!!!!!!!!!!!!!!!"`. next time you see garbage output from a quantized model, you'll know exactly where to start looking.

until next time, adios!
