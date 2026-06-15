---
layout: post
title: "FP8: how a number actually lives in 8 bits"
description: a from-first-principles walk through floating point - binary fractions, the mantissa, the implicit leading 1, exponent bias, and subnormals - building all the way up to the FP8 E4M3 format and why it maxes out at exactly 448 with no infinity.
tags: [Quantization, Deep Learning, Floating Point]
version: Released
release: 15-03-2026
---

so okay, i was writing this post about a nasty production bug - a multimodal model that just kinda starts spitting out garbage the second you flip on FP8 KV cache quantization. but halfway through writing it i hit this wall, i mean i literally couldn't explain the bug without first explaining how FP8 numbers actually work, like all the way down to the individual bits. and that explanation just kept growing and growing until, yeah, it was clearly its own post. so here we are.

so this is that foundation. by the end you'll know exactly how a number actually lives inside 8 bits - the mantissa, the exponent, the sneaky implicit 1, the subnormal range near zero, the full FP8 E4M3 format and all. we'll build it up slowly from binary fractions, and honestly a bunch of design choices that look totally bizarre at first start making complete sense as you go (like why the biggest number it can possibly hold is exactly 448, and why it has literally no way to even write infinity, weird right?).

and yeah, i know - "floating point fundamentals" sounds like the most boring possible topic, uggh, trust me i get it. but stick with it, because in the next post every single piece of this comes right back to explain a real failure where one mis-set number kinda snowballs into a model that's completely lost its mind. and you can't really appreciate that collapse until you know how the format is *supposed* to behave when nothing's wrong, right? so, the basics.

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

and this bucket thing gets worse as numbers get bigger. near zero, your buckets might be 0.00195 apart (subnormal range). but near the max value of 448, the gap between adjacent representable numbers is 32 (the difference between 416 and 448). so a value like 430 gets rounded to either 416 or 448 - you lose up to 14 in either direction. same 8 buckets per exponent range, but the buckets themselves get wider as the numbers get larger. keep this one in your back pocket - those widening buckets near 448 are exactly where things fall apart in the next post.

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

so stored value 7 maps to 2⁰ = 1.0, which sits right in the middle. below 7 you get exponents that shrink numbers (2⁻¹ = 0.5, 2⁻⁶ = 0.015 - still positive, just tiny!), above 7 you get exponents that grow them (2¹ = 2, 2⁸ = 256). it's not about tiny numbers - it's about whether the exponent makes your number bigger or smaller than 1.

why specifically 7 and not 8 or 6? it comes from the formula bias = 2^(bits-1) - 1. for 4 exponent bits: 2³ - 1 = 7. same logic everywhere - FP16 with 5 exponent bits uses bias 15, FP32 with 8 exponent bits uses bias 127. it just gives you a nice balanced split between the "bigger than 1" and "smaller than 1" side of things.

and just to really drive this home - what if we didn't use a bias at all? then our stored exponent values 0 to 15 would just directly mean 2⁰ to 2¹⁵. the smallest exponent would be 2⁰ = 1, and everything we could represent would be ≥ 1.0. we'd have a much bigger range on the top end (up to 2¹⁵ = 32,768!) but we'd completely lose the ability to represent anything between 0 and 1. no 0.5, no 0.1, no 0.015 - none of it. for something like the small fractional activations that show up all over a neural net, that would be useless. the bias is what gives us that whole sub-1.0 world by letting some exponents become negative (like -6), which makes the *result* a tiny positive number (2⁻⁶ = 0.015), not a negative one.

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

subnormals give us extra buckets near zero. pretty important when you're doing math with tiny numbers, which... yeah, comes up a lot in neural nets.

## FP8 E4M3: cramming a number into one byte

alright, so now let's just put it all together and actually understand FP8 E4M3 properly. this is the exact format that, in the next post, quietly breaks a whole production model - so getting it down here is basically what'll make that bug make sense later, i promise.

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

8 bits. that's it. one byte per value compared to FP16's 2 bytes - 50% memory reduction. sounds great, right? well... yes, but there's a catch. actually there are several catches.

### why is the max value exactly 448?

okay this is where it gets kinda fun. let me just walk you through it step by step, because this number turns out to be weirdly crucial later.

**step 1: biggest exponent we can use**

4 exponent bits can store 0000 to 1111 (0 to 15). as we discussed earlier, the bias of 7 lets us split this into tiny positive numbers (exponents below 7 shrink things below 1.0) and big numbers (exponents above 7 grow things). you subtract 7 from the stored value to get the actual exponent:

```
stored 1  → 1 - 7 = -6 → 2^-6 = 0.015  (smallest normal; 0000 is reserved for subnormals)
stored 7  → 7 - 7 =  0 → 2^0  = 1.0    (middle ground)
stored 15 → 15 - 7 = 8 → 2^8  = 256    (big number)
```

so max exponent = 8, meaning 2⁸ = 256.

**step 2: biggest mantissa we can use**

3 mantissa bits give us 8 possible values (000 to 111). with the implicit leading 1, mantissa `110` gives us 1.75. but wait - what about `111` which would give 1.875?

here's the catch - at this top exponent, the all-ones pattern (exponent `1111` *and* mantissa `111`) is the slot reserved for NaN (Not a Number) - the value that means "this calculation went wrong," like 0/0 or sqrt(-1). so we can't use mantissa `111` at the very top; the biggest usable mantissa up here is `110` = 1.75.

(important: `111` is perfectly fine at *lower* exponents - e.g. 2⁷ × 1.875 = 240 is a totally real number. it's only the all-ones-everything pattern that means NaN. we'll see this again in a second.)

**step 3: multiply them together**

```
max value = 2^8 × 1.75
          = 256 × 1.75
          = 448 ✓
```

if that top mantissa slot weren't reserved for NaN, max would be 256 × 1.875 = 480. we lose 32 of range just to have a way to signal NaN. worth it? absolutely - NaN is how the hardware tells you a computation went off the rails, and you really do want to know that.

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
  exponent = 1111, mantissa = 111  →  NaN (the only special value!)
```

trade-off: no way to represent ∞, but we get 7 extra usable numbers (256-448). when you only have 256 total bit patterns, every one counts. and notice - this is the same point from step 2: across the *entire* format, the single all-ones pattern `1111` + `111` is the *only* thing that means NaN. every other bit pattern is a real, usable number.

### E4M3 vs E5M2: two flavors of FP8

quick aside - there's another FP8 format called E5M2 with 5 exponent bits and 2 mantissa bits:

| format | exponent | mantissa | max value | use case |
|--------|----------|----------|-----------|----------|
| E4M3 | 4 bits | 3 bits | 448 | inference (better precision) |
| E5M2 | 5 bits | 2 bits | 57,344 | training (larger range) |

E4M3 gives you more precision (8 mantissa levels vs 4), E5M2 gives you more range (max 57,344 vs 448). for kv-cache quantization during inference, E4M3 is the standard choice because precision matters more than range... usually. when it doesn't - well, that's a story for the next post.

## wrapping up

okay so that's basically the whole foundation. let me boil it down to the handful of things actually worth carrying forward:

1. **a float is just `sign × 2^exponent × 1.mantissa`** - the mantissa holds your actual digits (as buckets, not exact values), and the exponent slides the binary point around to place them.

2. **the implicit leading 1 is a free bit** - since normalized binary always starts with 1, we don't bother storing it. 3 stored mantissa bits actually buy you 4 bits of precision.

3. **the bias lets exponents go negative** - subtract 7 from the stored exponent so you can represent both tiny fractions (2⁻⁶) and big numbers (2⁸), instead of being stuck at ≥ 1.0.

4. **subnormals fill the gap near zero** - when the exponent bits are all zero, drop the implicit 1, giving a smooth ramp down to actual zero instead of a cliff.

5. **FP8 E4M3 maxes out at 448** - that's 2⁸ × 1.75, and it has no infinity (the designers traded ∞ for 7 more usable numbers). the single all-ones pattern is the only NaN.

8 bits. 256 possible values. that's literally the entire toolkit you get when you quantize a model's KV cache to FP8. and honestly it's a surprisingly sharp tool - right up until the thing you're trying to measure grows past 448, and then the whole measurement just kinda quietly collapses.

which is exactly what we'll do to it in the next post, i mean we take this exact format, drop it into vLLM, point it at a multimodal model, and just watch one mis-set "ruler" lobotomize the whole thing - no crash, no error, nothing, just confidently wrong output. anyway, see you there!
