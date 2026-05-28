---
title: "gradient clipping"
source: obsidian
---

Gradient clipping is a training stability technique used to control exploding gradients. The idea is simple: before the optimizer updates the weights, check how large the gradients are. If they are too large, shrink them down to a safe size. This does not stop the model from learning; it stops one unstable batch or time step from creating a massive update that damages the weights.

The most common version is clipping by norm. You compute the overall gradient norm across the model, and if it is larger than some threshold, you rescale the gradients so the norm becomes that threshold. For example, if the gradient norm is `100` and the max allowed norm is `1`, all gradients are scaled down by roughly `1/100`. The direction of the update stays mostly the same, but the step size becomes controlled.

The intuition is like putting a speed limit on learning. Without clipping, an exploding gradient can make the optimizer jump far away from a good region of the loss landscape. With clipping, the optimizer still moves in the gradient direction, but it cannot take a dangerously large step.

In PyTorch, this usually looks like:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This is usually called after `loss.backward()` and before `optimizer.step()`. The backward pass computes the gradients, clipping limits them, and then the optimizer uses the clipped gradients to update the weights.

Gradient clipping is especially common in RNNs, LSTMs, GRUs, sequence models, reinforcement learning, and any setup where training can become unstable. It does not solve every training problem, but it is one of the simplest protections against exploding gradients turning a run into `inf` or `nan`.
