---
layout: post
title: FP8, KV Cache Quantization, and the "!!!!" Bug
description: a deep dive into floating point fundamentals, FP8 quantization, and how a subtle scale miscalibration in vLLM's KV cache causes garbage output - tracing the full causal chain from bit patterns to NaN propagation.
tags: [vLLM, Quantization, Inference, Deep Learning]
version: Released
release: 08-02-2026
---

so at the end of my [attention post](/2026/01/22/attention-is-all-you-need-guys/), i teased about kv-cache and inference optimizations. well, here's that follow up - except it turned into something way bigger than i planned. what started as "let me explain kv-cache quantization" became a full investigation into a production bug where a multimodal model just... starts outputting `"!!!!!!!!!!!!!!!!"`. yeah, literally exclamation marks. nothing else.

to actually understand *why* that happens, we need to go deep - like, all the way down to individual bits in a floating point number. i know that sounds excessive, but i promise every piece connects. we'll build up from binary fractions to FP8 format to quantization scales, and then watch the whole thing collapse like dominoes when one design decision goes wrong.

fair warning - this is a long one and it's dense. the first half covers floating point fundamentals (how numbers actually live in 8 bits), and the second half traces the vLLM bug from root cause to garbage output. if you need to split it up, the transition between those two halves is a natural stopping point.

## part 1: floating point fundamentals

okay so you know how decimal works, right? digits after the decimal point represent powers of 10:

```
  0 . 3   5   7
      ↓   ↓   ↓
     1/10 1/100 1/1000
     10⁻¹ 10⁻² 10⁻³

0.357 = 3×(1/10) + 5×(1/100) + 7×(1/1000)
      = 0.3 + 0.05 + 0.007
```

each position is the previous position ÷ 10. simple enough.

well binary works the exact same way, just with base 2 instead of base 10:

```
  0 . 1   0   1
      ↓   ↓   ↓
     1/2  1/4  1/8
     2⁻¹  2⁻²  2⁻³

0.101 (binary) = 1×(1/2) + 0×(1/4) + 1×(1/8)
               = 0.5 + 0 + 0.125
               = 0.625 (decimal)
```

each position is the previous position ÷ 2. that's literally why it's 1/2, 1/4, 1/8 - they're just 2⁻¹, 2⁻², 2⁻³. nothing magical about it. just like decimal uses powers of 10, binary uses powers of 2.

### the mantissa (where your actual digits live)

the mantissa (some people call it "significand") holds the precision bits - the actual digits of your number. think of it as the "meat" of the number, while the exponent just tells you where to put the decimal point.

in FP8 E4M3 (which we'll be using throughout this post), we have 3 mantissa bits. each bit represents a fractional position:

```
1st bit → 1/2  (0.5)
2nd bit → 1/4  (0.25)
3rd bit → 1/8  (0.125)
```

so if your mantissa bits are `101`, you get: 1×(1/2) + 0×(1/4) + 1×(1/8) = 0.5 + 0 + 0.125 = 0.625

here's all 8 possible mantissa values:

| bits | calculation | fraction |
|------|-------------|----------|
| 000 | 0/2 + 0/4 + 0/8 | 0.000 |
| 001 | 0/2 + 0/4 + 1/8 | 0.125 |
| 010 | 0/2 + 1/4 + 0/8 | 0.250 |
| 011 | 0/2 + 1/4 + 1/8 | 0.375 |
| 100 | 1/2 + 0/4 + 0/8 | 0.500 |
| 101 | 1/2 + 0/4 + 1/8 | 0.625 |
| 110 | 1/2 + 1/4 + 0/8 | 0.750 |
| 111 | 1/2 + 1/4 + 1/8 | 0.875 |

with 3 bits, you get 8 levels of precision. that's... not a lot. but hey, we're cramming a number into 8 bits total, so sacrifices must be made.

and here's the thing people don't always get right away - these aren't like integers where every value has its own exact slot. these 8 values act more like **buckets**. if your actual number is, say, 0.3, well tough luck - there's no bucket for 0.3. your closest options are 0.250 and 0.375, and 0.3 gets rounded to whichever is nearest (0.250 in this case). the number 0.3 just doesn't exist in this format, it gets snapped to the nearest bucket.

think of it like a ruler that only has marks every 0.125 apart. you can't measure 0.3 exactly - you'd say "it's about at the 0.25 mark." the more mantissa bits you have, the finer the markings on your ruler. FP16 has 10 mantissa bits (1024 buckets per exponent range), FP32 has 23 (over 8 million buckets). we have 3 bits. 8 buckets. that's the precision tax you pay for squeezing a number into a single byte.

and this bucket thing gets worse as numbers get bigger. near zero, your buckets might be 0.00195 apart (subnormal range). but near the max value of 448, the gap between adjacent representable numbers is 32 (the difference between 416 and 448). so a value like 430 gets rounded to either 416 or 448 - you lose up to 14 in either direction. same 8 buckets per exponent range, but the buckets themselves get wider as the numbers get larger. keep this in mind - it matters when we get to the quantization part.

### the implicit leading 1 (free bit hack)

okay so here's a clever trick that gives us a free extra bit of precision. to understand it, let's think about scientific notation first.

in decimal, when you write a number in scientific notation, you always normalize it so there's exactly one non-zero digit before the decimal point:

```
1234   → 1.234 × 10³
0.0056 → 5.6 × 10⁻³
42.7   → 4.27 × 10¹
```

you'd never write `0.001234 × 10⁶` or `1234.0 × 10⁰` - you always shift the decimal point until you have one non-zero digit up front.

binary does the same thing. you shift the binary point until you have a 1 before it:

```
11 (decimal) = 1011 (binary)
             = 1.011 × 2³    (shifted right 3 times)

0.125 (decimal) = 0.001 (binary)
                = 1.0 × 2⁻³  (shifted left 3 times)
```

but here's what makes binary special - in decimal, that leading digit could be 1, 2, 3... up to 9. you have to actually store it because it could be any of those. but in binary? the only digits are 0 and 1. and since we shift until the leading digit isn't zero, it can **only** be 1. there's literally no other option.

so why store something that's always 1? we don't! we just agree that there's a 1 there and don't waste a bit on it:

```
what we STORE in the mantissa:  0 1 1   (3 bits)
what we actually MEAN:        1.0 1 1   (implicitly 1.011)
                              ↑
                              this 1 is free! never stored.
```

this means our 3 mantissa bits are actually giving us 4 digits of precision - the implicit 1 plus the 3 stored bits. for free. when you only have 8 bits total, getting one extra bit of precision without spending any storage is a pretty big deal.

so remember those 8 mantissa buckets from earlier (0.000 to 0.875)? with the implicit leading 1, they become 1.000 to 1.875. the mantissa value is always at least 1.0 and at most 1.875. the exponent then scales this up or down to wherever you need it.

the full formula for a normal number:

```
value = (-1)^sign × 2^(exponent - bias) × 1.mantissa
                                          ↑
                                    the free 1!

example: sign=0, exponent=1001 (9), mantissa=101
  = 1 × 2^(9-7) × 1.625
  = 1 × 4 × 1.625
  = 6.5
```

now quick thing - why is the bias 7? well, our exponent field can only store 0 to 15 (4 bits, all unsigned). but we want to represent both big numbers (like 256) and tiny numbers (like 0.015). the bias is just a trick to get both: instead of using the stored value directly, you subtract 7 from it.

```
stored 1  → 1 - 7 = -6  → 2⁻⁶ = 0.015  (tiny positive number)
stored 7  → 7 - 7 =  0  → 2⁰  = 1.0    (one, the middle)
stored 14 → 14 - 7 = 7  → 2⁷  = 128    (big number)
stored 15 → 15 - 7 = 8  → 2⁸  = 256    (biggest)
```

so stored value 7 maps to 2⁰ = 1.0, which sits right in the middle. below 7 you get exponents that shrink numbers (2⁻¹ = 0.5, 2⁻⁶ = 0.015 — still positive, just tiny!), above 7 you get exponents that grow them (2¹ = 2, 2⁸ = 256). it's not about tiny numbers - it's about whether the exponent makes your number bigger or smaller than 1.

why specifically 7 and not 8 or 6? it comes from the formula bias = 2^(bits-1) - 1. for 4 exponent bits: 2³ - 1 = 7. same logic everywhere - FP16 with 5 exponent bits uses bias 15, FP32 with 8 exponent bits uses bias 127. it just gives you a nice balanced split between the "bigger than 1" and "smaller than 1" side of things.

and just to really drive this home - what if we didn't use a bias at all? then our stored exponent values 0 to 15 would just directly mean 2⁰ to 2¹⁵. the smallest exponent would be 2⁰ = 1, and everything we could represent would be ≥ 1.0. we'd have a much bigger range on the top end (up to 2¹⁵ = 32,768!) but we'd completely lose the ability to represent anything between 0 and 1. no 0.5, no 0.1, no 0.015 - none of it. for something like kv-cache values where activations can be small fractions, that would be useless. the bias is what gives us that whole sub-1.0 world by letting some exponents become negative (like -6), which makes the *result* a tiny positive number (2⁻⁶ = 0.015), not a negative one.

### subnormal numbers (filling the gap near zero)

okay so there's a problem. with normal numbers and the implicit leading 1, the smallest positive number we can represent is:

```
smallest exponent = 0001 (stored) → actual = 1-7 = -6
smallest mantissa = 000 → 1.000 (with implicit 1)

smallest normal = 2^-6 × 1.0 = 0.015625
```

that leaves a huge gap between 0 and 0.015625. you literally can't represent 0.001 or 0.005 or 0.01!

```
without subnormals:
   0 ──────────────────────────── 0.015625 ────→
     ↑                               ↑
     zero                        smallest normal

     can't represent anything in between!
```

the fix is subnormal mode. when the exponent bits are all zeros (`0000`), special rules kick in:

1. the exponent is fixed at 2⁻⁶ (doesn't change)
2. the implicit leading 1 becomes a 0 instead (so it's `0.mantissa` not `1.mantissa`)

now why 2⁻⁶ specifically? it's not a random choice - it's the same exponent as the smallest normal number. remember, the smallest normal has stored exponent `0001`, which gives actual exponent 1 - 7 = -6, so it equals 2⁻⁶ × 1.0 = 0.015625. subnormals use that exact same 2⁻⁶, but since they drop the implicit leading 1 (using 0.mantissa instead of 1.mantissa), they produce values *below* 0.015625. this is the key insight - by sharing the same power of 2, subnormals extend smoothly downward from exactly where normals start.

this gives us 7 tiny values between 0 and the smallest normal number (and mantissa `000` with exponent `0000`? that's just plain old zero - 0.0 × 2⁻⁶ = 0):

| mantissa | 0.mantissa | value |
|----------|------------|-------|
| 000 | 0.0 | 0.0 × 2⁻⁶ = **0** (zero itself!) |
| 001 | 0.125 | 0.125 × 2⁻⁶ = **0.00195** |
| 010 | 0.25 | 0.25 × 2⁻⁶ = **0.00391** |
| 011 | 0.375 | 0.375 × 2⁻⁶ = **0.00586** |
| 100 | 0.5 | 0.5 × 2⁻⁶ = **0.00781** |
| 101 | 0.625 | 0.625 × 2⁻⁶ = **0.00977** |
| 110 | 0.75 | 0.75 × 2⁻⁶ = **0.01172** |
| 111 | 0.875 | 0.875 × 2⁻⁶ = **0.01367** |

look at how nicely this stitches together. the largest subnormal is 0.875 × 2⁻⁶ = 0.01367, and the smallest normal is 1.0 × 2⁻⁶ = 0.01563. the gap between them is only 0.00195 - which is the same step size as between any two adjacent subnormals! there's no sudden jump, no weird discontinuity. it's a perfectly smooth transition from subnormal territory into normal territory. if they'd picked 2⁻⁷ instead, the largest subnormal would be 0.00684 and the smallest normal would still be 0.01563 - that's a gap of 0.00879, way bigger than the subnormal step size. the numbers would "jump" at the boundary and the whole point of having subnormals would be undermined.

why does this matter? without subnormals, you get "sudden underflow":

```
a = 0.012
b = 0.010
a - b = 0.002 → UNDERFLOWS TO 0! (wrong!)

with subnormals:
a - b = 0.002 → stored as subnormal (correct!)
```

subnormals give us extra buckets near zero. pretty important when you're doing math with tiny numbers, which... yeah, we'll see why that matters soon.

## FP8 E4M3: the format that started the whole mess

alright, now let's put it all together and understand FP8 E4M3 properly. this is the specific format vLLM uses for KV cache quantization, and understanding it is key to understanding the bug.

```
FP8 E4M3 format:
┌───┬─────────┬─────────┐
│ S │ E E E E │ M M M   │
└───┴─────────┴─────────┘
  1      4         3     = 8 bits total

S = sign bit (1 bit)
E = exponent bits (4 bits), bias = 7
M = mantissa bits (3 bits)
```

8 bits. that's it. one byte per value compared to FP16's 2 bytes - 50% memory reduction. sounds great for kv-cache, right? well... yes, but there's a catch. actually there are several catches.

### why is the max value exactly 448?

this is where it gets fun. let me walk you through it step by step, because this number is crucial to understanding the bug later.

**step 1: biggest exponent we can use**

4 exponent bits can store 0000 to 1111 (0 to 15). as we discussed earlier, the bias of 7 lets us split this into tiny positive numbers (exponents below 7 shrink things below 1.0) and big numbers (exponents above 7 grow things). you subtract 7 from the stored value to get the actual exponent:

```
stored 0  → 0 - 7 = -7 → 2^-7 = 0.008  (tiny positive number)
stored 7  → 7 - 7 =  0 → 2^0  = 1.0    (middle ground)
stored 15 → 15 - 7 = 8 → 2^8  = 256    (big number)
```

so max exponent = 8, meaning 2⁸ = 256.

**step 2: biggest mantissa we can use**

3 mantissa bits give us 8 possible values (000 to 111). with the implicit leading 1, mantissa `110` gives us 1.75. but wait - what about `111` which would give 1.875?

here's the thing - `111` is **reserved for NaN** (Not a Number). we need a way to say "this calculation went wrong" (like 0/0 or sqrt(-1)). so the last mantissa pattern is sacrificed for that purpose.

**step 3: multiply them together**

```
max value = 2^8 × 1.75
          = 256 × 1.75
          = 448 ✓
```

if `111` wasn't reserved for NaN, max would be 256 × 1.875 = 480. we lose 32 numbers of range just to have NaN. worth it? absolutely - you'll see why when NaN starts appearing in our bug trace.

### FP8 E4M3 has NO infinity!

this one surprised me when i first learned about it. in standard IEEE formats (FP16/FP32), there's a special bit pattern for infinity:

```
FP16: S.11111.0000000000 = ±∞
FP32: S.11111111.00000000000000000000000 = ±∞
```

but FP8 E4M3 designers made a different choice. when the exponent is `1111`:

```
standard IEEE:
  exponent = 1111, mantissa = 000  →  ±Infinity
  exponent = 1111, mantissa ≠ 000  →  NaN

FP8 E4M3 (different!):
  exponent = 1111, mantissa = 000  →  256 (normal number!)
  exponent = 1111, mantissa = 001  →  288 (normal number!)
  exponent = 1111, mantissa = 010  →  320 (normal number!)
  ...
  exponent = 1111, mantissa = 110  →  448 (normal number! max!)
  exponent = 1111, mantissa = 111  →  NaN (only special value!)
```

trade-off: no way to represent ∞, but we get 7 extra usable numbers (256-448). when you only have 256 total bit patterns, every one counts.

### E4M3 vs E5M2: two flavors of FP8

quick aside - there's another FP8 format called E5M2 with 5 exponent bits and 2 mantissa bits:

| format | exponent | mantissa | max value | use case |
|--------|----------|----------|-----------|----------|
| E4M3 | 4 bits | 3 bits | 448 | inference (better precision) |
| E5M2 | 5 bits | 2 bits | 57,344 | training (larger range) |

E4M3 gives you more precision (8 mantissa levels vs 4), E5M2 gives you more range (max 57,344 vs 448). for kv-cache quantization during inference, E4M3 is the standard choice because precision matters more than range... usually. when it doesn't, well, that's our bug.

## part 2: the vLLM KV cache bug

alright, now we have all the building blocks. let's trace this bug from the beginning.

### the problem

when running Qwen3-VL (a multimodal model that handles both text and images) with FP8 KV cache quantization in vLLM:
- initial requests may work fine
- at some point, **all subsequent requests start failing** - both text AND image
- output becomes: `"!!!!!!!!!!!!!!!!!!!!!!"`

the configuration that triggered it:

```yaml
args: [
  '--kv-cache-dtype', 'fp8',
  '--calculate-kv-scales',
  # ... other args
]
```

two flags. that's all it took to create a cascading failure. let me show you exactly why.

### vLLM's scale calculation: the root cause

so here's the deal with FP8 quantization - you can't just shove FP16 values into FP8 directly. you need a **scale factor** to map the range of your actual values into the representable range of FP8 (which maxes out at 448, remember?).

vLLM computes these scales dynamically during the first forward pass:

```python
# from vllm/attention/layer.py
def calc_kv_scales(self, key: torch.Tensor, value: torch.Tensor):
    if self.calculate_kv_scales:
        # compute scale based on current values
        # ...
```

the scale constants from the environment:

```python
# from vllm/envs.py
Q_SCALE_CONSTANT: int = 200
K_SCALE_CONSTANT: int = 200
V_SCALE_CONSTANT: int = 100
```

the quantization and dequantization is straightforward:

```
quantization (storing to cache):
  fp8_value = original_value / k_scale

dequantization (reading from cache):
  original_value = fp8_value × k_scale
```

seems reasonable so far. but here's the critical issue - after the first forward pass:

```python
self.calculate_kv_scales = False  # never recalculate!
```

**the scale computed on the first request is used for ALL subsequent requests, regardless of their value distributions.** let that sink in for a second.

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
  severe precision loss (most bits wasted)
```

the first scenario (clipping) is what causes our bug. the second scenario (precision loss) would cause subtle quality degradation but not catastrophic failure. let's trace the catastrophic one.

### the clipping: where things start going wrong

the clipping happens at the hardware level. NVIDIA's CUDA instruction uses `__NV_SATFINITE` mode, which automatically clips values exceeding the FP8 range to 448 (the max). it's a safety mechanism - but in this case, "safety" creates a much worse problem.

here's what the error looks like at different value levels. the error formula is simple - how far off is the retrieved value from the original, as a percentage:

```
error = (original - retrieved) / original × 100

example: original = 50.0, retrieved = 11.2
  error = (50.0 - 11.2) / 50.0 × 100 = 77.6%
```

| original value | scaled (÷0.025) | after clipping | retrieved (×0.025) | error |
|----------------|-----------------|----------------|-------------------|-------|
| 5.0 | 200 | 200 | 5.0 | 0% |
| 11.2 | 448 | 448 | 11.2 | 0% |
| 15.0 | 600 | 448 | 11.2 | **(15-11.2)/15 = 25.3%** |
| 20.0 | 800 | 448 | 11.2 | **(20-11.2)/20 = 44.0%** |
| 50.0 | 2000 | 448 | 11.2 | **(50-11.2)/50 = 77.6%** |

anything above 11.2 gets clamped to exactly 11.2. now imagine you have an image with rich, diverse feature values and a bunch of them are above 11.2. they ALL become the same number. this is where it gets really bad.

### the identical values problem

when multiple tokens (especially image tokens in multimodal models) have K/V values above the clipping threshold, they ALL collapse to the same value:

```
original image token K values (diverse):
  [15.3, 18.7, 12.1, 19.5, 16.8, 20.0, 14.2, 17.9]

after FP8 quantization + clipping:
  [11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2]
```

that's... that's really bad. all diversity gone. all information gone. just a flat line of identical values. and this feeds right into the next failure.

### layernorm variance collapse: the NaN factory

remember layernorm from my [attention post](/2026/01/22/attention-is-all-you-need-guys/)? let me refresh. layernorm normalizes each value in a token's embedding individually, using this formula:

```
LayerNorm(x) = (x - μ) / sqrt(σ² + ε)

where:
  μ = mean of x
  σ² = variance of x
  ε = small constant (typically 1e-12)
```

so for a token embedding like [a, b, c, d], it computes μ and σ² across those values, then normalizes each one individually:

```
output[0] = (a - μ) / sqrt(σ² + ε)
output[1] = (b - μ) / sqrt(σ² + ε)
output[2] = (c - μ) / sqrt(σ² + ε)
output[3] = (d - μ) / sqrt(σ² + ε)
```

every element gets the same μ and σ², but each one is normalized separately. this matters because if σ² goes bad, *every single element* in that token's embedding blows up.

now watch what happens when you feed it those identical values:

```
x = [11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2]
μ = 11.2
σ² = Σ(x - μ)² / n = [0 + 0 + 0 + ...] / 8 = 0
```

variance is zero. okay, that's what the ε is for, right? to prevent division by zero? well... not exactly. here's where floating point strikes back.

in practice, GPUs use the efficient variance formula:

```
σ² = E[x²] - E[x]²
```

mathematically equivalent to the standard formula, but with floating point arithmetic, `E[x²]` and `E[x]²` are computed separately, and each has tiny rounding errors. when the true variance is zero (identical values), these rounding errors can actually make the result **slightly negative**:

```
11.2 is stored as 11.199999809265137 in FP32

with different accumulation order/rounding:
  σ² = -0.000000317835426  ← NEGATIVE!
```

and then:

```
denominator = sqrt(σ² + ε)
            = sqrt(-0.000000317835426 + 0.000000000001)
            = sqrt(-0.000000317834426)
            = NaN  ← square root of negative number!
```

boom. NaN. and once you have NaN, it's game over. because NaN is the most infectious value in all of computing.

### NaN propagation: one bad apple spoils the batch

this is the part that blew my mind when i first traced through it. a single NaN token corrupts EVERYTHING in one attention layer. let me show you exactly how.

**starting point:** let's say token 2 (an image token) got the identical values and produced NaN after layernorm. the other tokens are fine.

```
hidden states after LayerNorm:
  token 0: [0.5, -0.3, 0.8, 0.2, ...]     ← normal
  token 1: [0.1, 0.7, -0.4, 0.3, ...]     ← normal
  token 2: [NaN, NaN, NaN, NaN, ...]       ← corrupted!
  token 3: [-0.2, 0.4, 0.9, -0.5, ...]    ← normal
```

**Q/K/V projections:** token 2's NaN hidden states get multiplied by the weight matrices to compute Q, K, and V (remember from my [attention post](/2026/01/22/attention-is-all-you-need-guys/), each token's hidden state gets projected through learned weight matrices). here's what that looks like for the query projection:

```
Q₂ = hidden₂ @ W_q

[NaN, NaN, NaN, NaN] @ [[w₁₁, w₁₂, w₁₃],     [NaN, NaN, NaN]
                        [w₂₁, w₂₂, w₂₃],  =   
                        [w₃₁, w₃₂, w₃₃],      NaN × w₁₁ + NaN × w₂₁ + ...
                        [w₄₁, w₄₂, w₄₃]]      = NaN for EVERY output element
```

every output element involves summing products where at least one factor is NaN, so the entire result is NaN. same thing happens for K and V projections. so token 2's Q, K, and V are all completely NaN.

**attention scores (QK^T):** this is where it spreads. remember, attention scores are computed as Q @ K^T - every query dot-products with every key:

```
        k₀    k₁    k₂    k₃
                    ↑
                (all NaN)
    ┌────────────────────────
q₀  │  s₀₀   s₀₁   NaN   s₀₃ │
q₁  │  s₁₀   s₁₁   NaN   s₁₃ │
q₂  │  NaN   NaN   NaN   NaN │  ← entire row is NaN
q₃  │  s₃₀   s₃₁   NaN   s₃₃ │
    └────────────────────────
```

column 2 is NaN because k₂ is NaN. row 2 is NaN because q₂ is NaN. every other row has at least one NaN (from column 2).

**softmax: the point of no return**

```
softmax(x)_i = exp(x_i) / Σⱼ exp(x_j)
```

for row 0: [s₀₀, s₀₁, NaN, s₀₃]

```
exp([s₀₀, s₀₁, NaN, s₀₃]) = [exp(s₀₀), exp(s₀₁), NaN, exp(s₀₃)]

denominator = exp(s₀₀) + exp(s₀₁) + NaN + exp(s₀₃)
            = (some value) + NaN
            = NaN

softmax = [exp(s₀₀)/NaN, exp(s₀₁)/NaN, NaN/NaN, exp(s₀₃)/NaN]
        = [NaN, NaN, NaN, NaN]
```

**every single row** has a NaN in column 2, which means every row's softmax denominator is NaN, which means every attention weight becomes NaN. one corrupted token just made ALL attention weights NaN.

**attention output:** NaN weights × values = NaN for ALL tokens.

**residual connection:** even the tokens that had perfectly normal inputs are now dead:

```
token 0: [0.5, -0.3, 0.8, ...] + [NaN, NaN, NaN, ...]
       = [NaN, NaN, NaN, ...]  ← normal + NaN = NaN!
```

the whole thing is corrupted. in ONE layer. within a single attention computation, we went from 1/4 tokens having NaN (25%) to 4/4 tokens (100%). the remaining layers just propagate 100% NaN forward.

```
start of layer:     1/4 tokens NaN (25%) - just token 2
after QK^T:         row 2 and column 2 are NaN
after softmax:      ALL rows become NaN
after attention:    ALL tokens' outputs are NaN
after residual:     4/4 tokens NaN (100%)

corruption complete in ONE layer!
```

### why ALL future requests fail too

here's the truly nasty part. once NaN gets into the KV cache, it stays there. and because of how attention works - every new query attends to ALL cached keys - any new request will compute attention scores against those NaN keys.

```
new request arrives (perfectly valid text):
1. new_query dot NaN_cached_key = NaN
2. NaN in softmax denominator = NaN
3. NaN attention weights = NaN output

result: "!!!!!!!!!!!!!!!!!!!"
```

the model is effectively poisoned until you restart the whole inference server. both text requests and image requests fail. it doesn't matter how simple or valid the new input is - the cached NaN values will corrupt everything through the attention mechanism.

**but why "!" specifically?** okay so by this point, every single logit (the score the model gives to each possible next token) is NaN. the model needs to pick a token to output, so it does `argmax` over all those logits to find the "best" one.

but here's the thing about NaN - any comparison with NaN returns false. `NaN > 5.0`? false. `NaN > -1000.0`? false. `NaN > NaN`? also false. so when argmax scans through the list looking for the biggest value, it compares each element to its current best. since every comparison returns false, nothing ever beats the initial candidate. and what's the initial candidate? **index 0** - the very first element. that's just how PyTorch's argmax works when everything is NaN - it returns 0 by default.

so the model picks token ID 0. and in Qwen3-VL's tokenizer, token ID 0 happens to be `"!"`. that's it. it's not that the model "chose" exclamation marks - it's that NaN logits make argmax fall back to the first token in the vocabulary, and "!" is sitting right there at index 0.

(if this were a different model with a different tokenizer, you might see a different garbage character - whatever token sits at index 0 in *that* vocabulary. but the pattern is always the same: one token repeated forever.)

oh, and it gets even worse if **prefix caching** is enabled. vLLM can cache KV entries for common prompt prefixes so they don't need to be recomputed. if the NaN-corrupted KV entries get saved as a prefix cache entry, then *any future request with a matching prefix* will reuse those poisoned entries directly - without even going through the computation that would normally trigger the bug. the corruption essentially gets "baked in" to the cache. even without prefix caching though, the corruption persists within the running server's memory until restart.

### the complete causal chain

let me trace the whole thing end to end, because seeing the full chain really drives home how a tiny design decision cascades into a catastrophic failure:

```
1. --kv-cache-dtype fp8 + --calculate-kv-scales enabled
       ↓
2. first request arrives, scale computed and LOCKED
       ↓
3. subsequent request has K/V values outside calibrated range
       ↓
4. values clipped to FP8 max (448) during quantization
       ↓
5. many tokens get identical dequantized values
       ↓
6. LayerNorm variance → zero (or slightly negative due to FP arithmetic)
       ↓
7. sqrt(negative) = NaN
       ↓
8. NaN propagates through Q/K/V projections
       ↓
9. NaN in attention scores (QK^T column from corrupted key)
       ↓
10. softmax produces all-NaN weights (one NaN poisons entire sum)
       ↓
11. ALL tokens become NaN after residual connection
       ↓
12. NaN stored in KV cache
       ↓
13. ALL future requests attend to NaN keys → permanent corruption
       ↓
14. output: "!!!!!!!!!!!!!!!!!!!"
```

14 steps from a configuration flag to garbage output. and the fix is embarrassingly simple.

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

`--kv-cache-dtype auto` just stores the KV cache in its original precision (FP16 or BF16) without trying to squeeze it into FP8. no scales, no clipping, no NaN. yeah you use more memory (roughly 2× for the cache), but the model actually works correctly. for multimodal models where image tokens and text tokens have very different value ranges, this is really the only safe option right now.

could vLLM make FP8 work properly here? totally. a few ideas:
- **recalculate scales per request** - instead of locking them after the first pass, update them as new value ranges come in
- **per-token dynamic quantization** - give each token its own scale instead of one global scale for everything
- **outlier-aware quantization** - detect values that would clip and handle them separately

but as of now, with `--calculate-kv-scales`, the scale is computed once on the first request and locked forever. and that "forever" is what breaks everything.

## takeaways

a few things i want you to walk away with:

1. **FP8 E4M3 maxes out at 448** - that's 256 × 1.75. the 256 comes from 2⁸, where 8 is the max actual exponent (stored exponent 15, minus bias 7, gives 15 - 7 = 8). and 1.75 is the largest usable mantissa (1.110 in binary - remember, 111 is reserved for NaN).

2. **the scale is everything** - the scale is basically the ruler you use to fit your numbers into FP8's tiny range. if you set the ruler based on small numbers and then big numbers show up later, they all get chopped down to the same max value. that's clipping. and clipping means different numbers become identical, which is where everything falls apart.

3. **NaN is incredibly infectious** - one NaN value in one token propagates through the attention mechanism to corrupt ALL tokens in ONE layer. softmax is the amplifier - it can't handle even a single NaN in its input because the denominator sum becomes NaN.

4. **floating point arithmetic has subtle failure modes** - the variance of identical values should be zero mathematically, but floating point rounding can make it slightly negative, and sqrt(negative) = NaN. the gap between math and computation is where bugs hide.

5. **the KV cache persists across requests** - this is normally a feature (faster inference!), but when corruption gets in, it becomes a bug amplifier. NaN values in the cache poison all future requests until restart.

so yeah, that's how two command line flags, a one-time scale calculation, and the fundamental limitations of 8-bit floating point combine to produce `"!!!!!!!!!!!!!!!!!!!"`. pretty wild chain of events for something that started as a memory optimization.

next time you see garbage output from a quantized model, you'll know exactly where to start looking. until next time, adios!
