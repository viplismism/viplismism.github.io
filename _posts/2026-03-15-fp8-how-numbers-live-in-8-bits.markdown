---
layout: post
title: "FP8: how a number actually lives in 8 bits"
description: a from-first-principles walk through floating point - binary fractions, the mantissa, the implicit leading 1, exponent bias, and subnormals - building all the way up to the FP8 E4M3 format and why it maxes out at exactly 448 with no infinity.
tags: [Quantization, Deep Learning, Floating Point]
version: Released
release: 15-03-2026
---

![fp8-meme](/images/fp8-how-numbers-live-in-8-bits/fp8-meme.png)

so okay, i was writing this post about a nasty production bug - a multimodal model that just kinda starts spitting out garbage the second you flip on FP8 KV cache quantization. but halfway through, i hit this wall. i mean i literally couldn't explain the bug without first explaining how FP8 numbers actually work. like all the way down to the individual bits. and that explanation kept growing until yeah, it was clearly its own post. so here we are.

this is that foundation. by the end you'll know exactly how a number lives inside 8 bits - the mantissa, the exponent, the sneaky implicit 1, the subnormal range near zero, the whole FP8 E4M3 format. we'll build it up slowly from binary fractions, and honestly a bunch of design choices that look totally bizarre at first start making complete sense as you go.

i know - "floating point fundamentals" sounds like the most boring possible topic, uggh, i get it. but stick with it, because in the next post every single piece of this comes right back to explain a real failure where one mis-set number snowballs into a model that's completely lost its mind. you can't really appreciate that collapse until you know how the format is *supposed* to work, right? so. the basics.

## decimal, then binary

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

binary works the exact same way, just with base 2 instead of base 10:

```
0 . 1   0   1
    ↓   ↓   ↓
    1/2  1/4  1/8
    2⁻¹  2⁻²  2⁻³

0.101 (binary) = 1×(1/2) + 0×(1/4) + 1×(1/8)
               = 0.5 + 0 + 0.125
               = 0.625 (decimal)
```

each position is the previous position ÷ 2. that's literally why it's 1/2, 1/4, 1/8 - they're just 2⁻¹, 2⁻², 2⁻³. nothing magical. just like decimal uses powers of 10, binary uses powers of 2.

### the mantissa (where your actual digits live)

the mantissa (some people call it "significand") holds the precision bits - the actual digits of your number. think of it as the "meat" of the number. the exponent just tells you where to put the decimal point.

in FP8 E4M3 (which we'll use throughout), we have 3 mantissa bits. each bit represents a fractional position:

```
1st bit → 1/2  (0.5)
2nd bit → 1/4  (0.25)
3rd bit → 1/8  (0.125)
```

so if your mantissa bits are `101`, you get: 1×(1/2) + 0×(1/4) + 1×(1/8) = 0.5 + 0.125 = 0.625.

here's all 8 possible values:

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

3 bits, 8 levels. that's... not a lot. but we're cramming a number into 8 bits total, so sacrifices must be made.

and here's the thing people don't always get right away - these aren't like integers where every value has its own exact slot. these 8 values are more like **buckets**. if your actual number is 0.3? tough luck - there's no bucket for 0.3. your closest options are 0.250 and 0.375, and 0.3 gets snapped to whichever is nearest (0.250 in this case).

think of it like a ruler that only has marks every 0.125 apart. you can't measure 0.3 exactly - you'd say "it's about at the 0.25 mark." more mantissa bits = finer marks. FP16 has 10 mantissa bits (1024 buckets per range), FP32 has 23 (over 8 million). we have 3 bits. 8 buckets. that's the precision tax you pay for squeezing a number into a single byte.

one more thing worth keeping in mind: these buckets get *wider* as numbers get bigger. near zero, buckets might be 0.00195 apart. but near the max value of 448, adjacent representable numbers are 32 apart (416 and 448 are neighbors, nothing in between). so a value like 430 gets rounded to either 416 or 448 - you lose up to 14 in either direction. same 8 buckets per range, but the buckets themselves stretch. this matters a lot when we get to the quantization part in the next post.

### the implicit leading 1 (free bit hack)

okay here's a clever trick that gives us a free extra bit of precision. let's think about scientific notation first.

in decimal scientific notation, you always normalize so there's exactly one non-zero digit before the decimal:

```
1234   → 1.234 × 10³
0.0056 → 5.6 × 10⁻³
42.7   → 4.27 × 10¹
```

you'd never write `0.001234 × 10⁶` - you always shift until you have one non-zero digit up front.

binary does the same thing. you shift the binary point until you have a 1 before it:

```
11 (decimal) = 1011 (binary)
             = 1.011 × 2³    (shifted right 3 times)

0.125 (decimal) = 0.001 (binary)
                = 1.0 × 2⁻³  (shifted left 3 times)
```

here's what makes binary special though. in decimal, that leading digit could be 1 through 9, so you have to store it. but in binary? the only non-zero digit is 1. since we always shift until the leading digit isn't zero, it can **only ever be 1**. there's literally no other option.

so why store something that's always 1? we don't. we just agree it's there and never waste a bit on it:

```
what we STORE in the mantissa:  0 1 1   (3 bits)
what we actually MEAN:        1.0 1 1   (implicitly 1.011)
                              ↑
                              this 1 is free! never stored.
```

our 3 mantissa bits are actually giving us 4 digits of precision. the implicit 1 plus the 3 stored bits, for free. when you only have 8 bits total, that's a pretty big deal.

so those 8 mantissa buckets from earlier (0.000 to 0.875)? with the implicit leading 1, they become 1.000 to 1.875. the mantissa value is always at least 1.0. the exponent then scales this up or down to wherever you need it.

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

### the exponent bias (why subtract 7?)

quick thing - why is the bias 7? our exponent field stores values 0 to 15 (4 bits, all unsigned). but we want to represent both big numbers (like 256) and tiny numbers (like 0.015). the bias is the trick that buys us both: instead of using the stored value directly, you subtract 7 from it.

```
stored 1  → 1 - 7 = -6  → 2⁻⁶ = 0.015  (tiny positive number)
stored 7  → 7 - 7 =  0  → 2⁰  = 1.0    (one, the middle)
stored 14 → 14 - 7 = 7  → 2⁷  = 128    (big number)
stored 15 → 15 - 7 = 8  → 2⁸  = 256    (biggest)
```

stored value 7 maps to 2⁰ = 1.0, right in the middle. below 7 you get exponents that shrink numbers (2⁻¹ = 0.5, 2⁻⁶ = 0.015 - still positive, just tiny!), above 7 you get exponents that grow them. it's not about tiny numbers - it's about whether the exponent makes your number bigger or smaller than 1.

why 7 specifically? bias = 2^(bits-1) - 1. for 4 exponent bits: 2³ - 1 = 7. same logic everywhere - FP16 uses bias 15, FP32 uses bias 127. gives you a balanced split.

without a bias, stored exponents 0-15 would directly mean 2⁰ to 2¹⁵. the smallest exponent would be 1, so everything would be ≥ 1.0. huge range on the top end (up to 32,768!), but you'd completely lose the ability to represent anything between 0 and 1. no 0.5, no 0.1, none of it. for neural net activations that are often small fractions, that'd be useless. the bias is what gives us that whole sub-1.0 world.

### subnormal numbers (filling the gap near zero)

okay so there's a problem. with normal numbers and the implicit leading 1, the smallest positive number we can represent is:

```
smallest exponent = 0001 (stored) → actual = 1-7 = -6
smallest mantissa = 000 → 1.000

smallest normal = 2^-6 × 1.0 = 0.015625
```

that leaves a gap between 0 and 0.015625. you literally can't represent 0.001 or 0.005 or 0.01!

```
without subnormals:
0 ──────────────────────────── 0.015625 ────→
↑                               ↑
zero                        smallest normal

nothing in between!
```

the fix is subnormal mode. when the exponent bits are all zeros (`0000`), two special rules kick in:

1. the exponent is fixed at 2⁻⁶ (doesn't change)
2. the implicit leading 1 becomes a 0 (so it's `0.mantissa` not `1.mantissa`)

why 2⁻⁶ specifically? it's the same exponent as the smallest normal number - stored exponent `0001` gives actual exponent 1-7 = -6, so it's 2⁻⁶ × 1.0 = 0.015625. subnormals use that same 2⁻⁶, but since they drop the implicit 1, they produce values *below* 0.015625. same power of 2, smooth transition downward from exactly where normals start.

this gives us 7 tiny values between 0 and the smallest normal (mantissa `000` with exponent `0000` is just zero itself - 0.0 × 2⁻⁶ = 0):

| mantissa | 0.mantissa | value |
|----------|------------|-------|
| 000 | 0.0 | 0.0 × 2⁻⁶ = **0** |
| 001 | 0.125 | 0.125 × 2⁻⁶ = **0.00195** |
| 010 | 0.25 | 0.25 × 2⁻⁶ = **0.00391** |
| 011 | 0.375 | 0.375 × 2⁻⁶ = **0.00586** |
| 100 | 0.5 | 0.5 × 2⁻⁶ = **0.00781** |
| 101 | 0.625 | 0.625 × 2⁻⁶ = **0.00977** |
| 110 | 0.75 | 0.75 × 2⁻⁶ = **0.01172** |
| 111 | 0.875 | 0.875 × 2⁻⁶ = **0.01367** |

look at how nicely this stitches together. the largest subnormal is 0.01367, and the smallest normal is 0.01563. the gap between them is 0.00195 - same step size as between any two adjacent subnormals. no sudden jump, no weird discontinuity. if they'd picked 2⁻⁷ instead, the boundary gap would be way bigger than the subnormal step size and the whole smooth-ramp thing breaks.

why does it matter?

```
a = 0.012
b = 0.010
a - b = 0.002 → without subnormals: UNDERFLOWS TO 0! (wrong!)
              → with subnormals:    stored as subnormal (correct!)
```

extra buckets near zero. comes up a lot in neural nets.

## FP8 E4M3: cramming a number into one byte

alright, now let's put it all together. FP8 E4M3 is the exact format that quietly breaks a production model in the next post - so getting this down here is basically what'll make that bug make sense, i promise.

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

8 bits. one byte per value compared to FP16's 2 bytes - 50% memory reduction. sounds great, right? well... yes, but there are a few catches.

### why is the max value exactly 448?

kinda fun to derive this. let me walk through it step by step, because this number turns out to be weirdly crucial later.

**biggest exponent we can use**

4 exponent bits store 0000 to 1111 (0 to 15). with a bias of 7, subtract 7 from the stored value to get the actual exponent:

```
stored 1  → 1 - 7 = -6 → 2^-6 = 0.015  (smallest normal)
stored 7  → 7 - 7 =  0 → 2^0  = 1.0
stored 15 → 15 - 7 = 8 → 2^8  = 256    (biggest)
```

max exponent = 8, meaning 2⁸ = 256.

**biggest mantissa we can use**

3 mantissa bits give us values 000 to 111. with the implicit leading 1, mantissa `110` = 1.75. but what about `111` = 1.875?

here's the catch - at this top exponent, the all-ones exponent+mantissa pattern (exponent `1111` *and* mantissa `111`) is reserved for NaN. the sign bit can be 0 or 1, but either way that exponent+mantissa combo means "this calculation went wrong" - like 0/0 or sqrt(-1). so the biggest usable mantissa at the top is `110` = 1.75.

(important: `111` is totally fine at *lower* exponents. like 2⁷ × 1.875 = 240 is a perfectly real number. only the all-ones-everything pattern is special. more on this in a second.)

**put them together**

```
max value = 2^8 × 1.75
          = 256 × 1.75
          = 448 ✓
```

if that top slot weren't reserved for NaN, max would be 256 × 1.875 = 480. we lose 32 of range to have a way to signal "something went wrong." worth it? absolutely - you'll see exactly why in the next post.

### FP8 E4M3 has NO infinity!

this one surprised me when i first learned it. in standard IEEE formats (FP16/FP32), there's a special pattern for infinity:

```
FP16: S.11111.0000000000 = ±∞
FP32: S.11111111.00000000000000000000000 = ±∞
```

FP8 E4M3 made a different call. when the exponent is `1111`:

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
  exponent = 1111, mantissa = 111  →  NaN (for either sign bit)
```

trade-off: no way to represent ∞, but 7 extra usable positive numbers (256-448), plus their negative counterparts. when you only have 256 total bit patterns, every one counts. and yeah - this confirms the point from above. the all-ones exponent+mantissa pattern `1111` + `111` is the only NaN pattern, ignoring sign. every other exponent+mantissa pattern is a real number.

### E4M3 vs E5M2: two flavors of FP8

quick aside - there's another FP8 format called E5M2 with 5 exponent bits and 2 mantissa bits:

| format | exponent | mantissa | max value | use case |
|--------|----------|----------|-----------|----------|
| E4M3 | 4 bits | 3 bits | 448 | inference (better precision) |
| E5M2 | 5 bits | 2 bits | 57,344 | training (larger range) |

E4M3 gives you more precision (8 mantissa levels vs 4), E5M2 gives you more range (max 57,344 vs 448). for kv-cache during inference, E4M3 is the standard choice because precision matters more than range... usually. when it doesn't, well, that's the next post.

and if you're wondering where 57,344 comes from - same derivation, different rules.

E5M2 has 5 exponent bits, so bias = 2⁴ - 1 = 15. max stored exponent is 11111 = 31, minus bias 15 = actual exponent 16. but 11111 is reserved - E5M2 follows standard IEEE here, meaning the entire exponent-all-ones row (11111 + *any* mantissa) means NaN or infinity. so the max usable exponent is 11110 = 30 - 15 = 15, giving 2¹⁵ = 32,768.

the mantissa? with 2 bits, max is `11` = 1.75. and here's the key difference - in E5M2, mantissa `11` is completely fine. there's no restriction on it. the reservation is on the whole exponent-11111 row, not a specific mantissa value. so:

```
E4M3: only the all-ones exponent+mantissa pattern (1111 AND 111) = NaN
      → exponent 1111 + mantissa 110 = 448 is still valid!

E5M2: entire all-ones-exponent row (11111 + anything) = NaN/inf
      → exponent 11110 + mantissa 11 = 57,344 is valid
```

2¹⁵ × 1.75 = 32,768 × 1.75 = **57,344**. same logic as 448, just bigger numbers and a different NaN rule.

## wrapping up

okay so that's basically the whole foundation. let me boil it down to the things worth carrying forward:

1. **a float is just `sign × 2^exponent × 1.mantissa`** - the mantissa holds your actual digits (as buckets, not exact values), and the exponent slides the binary point around to place them.

2. **the implicit leading 1 is a free bit** - normalized binary always starts with 1, so we don't store it. 3 mantissa bits buy you 4 bits of precision.

3. **the bias lets exponents go negative** - subtract 7 from the stored exponent so you can represent both tiny fractions (2⁻⁶) and big numbers (2⁸), instead of being stuck at ≥ 1.0.

4. **subnormals fill the gap near zero** - when exponent bits are all zero, drop the implicit 1, giving a smooth ramp down to actual zero instead of a cliff.

5. **FP8 E4M3 maxes out at 448** - that's 2⁸ × 1.75, with no infinity. the all-ones exponent+mantissa pattern is NaN, for either sign bit.

8 bits. 256 possible values. that's literally the entire toolkit when you quantize a model's KV cache to FP8. and honestly it's a surprisingly sharp tool - right up until the thing you're measuring grows past 448, and then the whole measurement just kinda quietly collapses.

which is exactly what we'll do to it in the next post. we take this exact format, drop it into vLLM, point it at a multimodal model, and watch one mis-set "ruler" lobotomize the whole thing - no crash, no error, nothing, just confidently wrong output. see you there!
