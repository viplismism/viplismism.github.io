---
title: "exploding gradients"
source: obsidian
---

Exploding gradients happen when the learning signal grows too large during backpropagation. In deep networks or recurrent networks, the gradient is passed backward through many layers or time steps, and at each step it gets multiplied by derivatives and weight matrices. If those repeated multiplications are mostly larger than 1, the gradient can grow very quickly instead of staying controlled.

A simple mental model is this: imagine the gradient starts at 1 and every step multiplies it by 2. Then the gradient becomes 1, 2, 4, 8, 16, 32, and keeps growing. In a real neural network this is happening with matrices, not just one number, but the intuition is the same. The backward signal gets amplified too much as it travels through the model.

This becomes dangerous because the optimizer uses the gradient to update the weights. With SGD, the update is basically `w = w - lr * grad`. If the learning rate is `0.001` but the gradient is `1,000,000`, the update is `1000`, which is massive for a neural network weight. A weight that was around `0.2` can suddenly become something like `-999.8`.

Once weights become huge, the next forward pass can produce huge activations and logits. Then operations like exponentials inside softmax can overflow. For example, `exp(1000000)` is too large to represent numerically, so it becomes `inf`. After that, operations like `inf - inf`, `0 * inf`, or `inf / inf` can produce `nan`. Once a `nan` appears in the loss, gradients, or weights, it usually spreads through the whole training run.

In practice, exploding gradients often look like a training run where the loss is normal for a while, then suddenly spikes hard, then becomes `nan`. You might see something like loss `2.8`, then `45`, then `1000000`, then `nan`. The model has not just become worse; the numerical computation has become unstable.

Common fixes include lowering the learning rate, using better initialization, using normalization layers, adding residual connections, using gated architectures in sequence models, and applying [gradient clipping](/glossary/gradient-clipping). Gradient clipping is the most direct fix because it limits how large the gradient is allowed to be before the optimizer updates the weights.
