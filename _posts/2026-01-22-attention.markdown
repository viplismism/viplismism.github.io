---
layout: post
title: Attention is all you need guys
description: a deep dive into transformers and the attention mechanism from first principles - covering positional embeddings, Q/K/V projections, multi-head attention, residual connections, layer normalization, and feed-forward networks.
tags: [Transformers, Deep Learning, NLP]
version: Released
release: 22-01-2026
---

**heads up!** this is a long one - around 6,000 words and 35k characters. grab a coffee (or two), find a quiet spot, and give it your undivided attention (pun intended), prefer to read it on a laptop or a big screen! 

yeah, I know what you're thinking - "I'll just throw this at Claude and get the summary." sure, you could do that. but if you actually want to *understand* attention from first principles, there's no shortcut. trust me, the 30 minutes you spend here will save you hours of confusion later. your call though!

## here we go!

i know there are ton of explanations of the transformers on the web but after reading it for 18th time (yeah, I read that "attention" paper back in my 2nd year) and until recently, I quite got all the gist of it, so i mean i think it's a good time to put that into words here.

tldr : this might be the only explanation you will ever need to actually understand attention from first principles, and i hope you have some idea of rnns, lstms and a bit of how the gradient descent works, yeah the og algo, if not, would suggest you to just have a quick read (yeah, spin your claude now!). ok so.. 

### the sequence problem or why rnns kinda suck

okay so let's rewind to like 2015-ish?! back then, if you wanted to do anything with text or sequential data, rnns were pretty much your only real option. i mean the core idea was actually super intuitive when you think about it.

so imagine you're reading a really long paragraph word by word, and you're trying to remember everything as you go along. you start with something like "the cat that was sitting on the mat that john bought last week..." and by the time you get to the end of the sentence, you've kinda forgotten what was at the beginning, right? well that's basically what happened with rnns (recurrent neural networks) - the old school way of processing text before transformers.

rnns would process sentences one word at a time, maintaining this "mental note" about everything they'd seen so far. sounds reasonable, right? except they had this massive problem: the further back in the sentence you went, the more the information would just fade away. it's because of this thing called vanishing gradients, yeah the og problem.

![Vanishing Gradients](/videos/attention-is-all-you-need-guys/VanishingGradients.gif)

basically during backpropagation (when the model weights are updated), the error signal has to travel backwards through time to update all those weights. the issue is that the gradients get multiplied at each timestep, and since these values are typically fractions, they shrink crazy fast.

just to make it more clear, imagine you have a sequence of 100 words. now at each step back, your gradient gets multiplied by something like a very small number, let's say 0.9. by the time you reach the beginning, you're looking at 0.9^100 ≈ 0.000027 - that's practically zero dude! now if you want to update the weights by this much amount, it's basically no update at all.

this is why rnns struggled with longer contexts. if something important happened 50 steps ago, the model barely updates those early weights because the gradient signal is too weak by the time it travels back. the connections between those distant parts of the sequence become nearly impossible to learn. and that's where the og thing came, the transformers baby.

### well what really is a transformer?!

keeping things straight, transformer is basically a machine learning model which does sequence modelling. what it means is when you have a sequence of things and you feed the model some first x-1 units of that sequence, it tries to predict what the xth unit should be. this is it. this works for literally any type of sequence data. you can feed a transformer text, music notes, protein structures, code - anything where order matters and patterns exist.

like music chords, where after seeing [C, F, G, Am], the model might predict the next chord is F again because that's a common progression.

or in something like "I am from India and I am..." and the model predicted "Indian", where all the words before "Indian" are those x-1 number of units and the xth unit was "Indian" which the model predicted! unlike rnns, transformers can easily look at the entire context at once, so they don't forget what happened way back at the beginning.

now if you're reading this as someone who doesn't know ml (dude, learn ml bro, duh!), let me quickly explain how words actually get fed into these models. so there's this process called tokenization, which is just a fancy way of saying "converting words to numbers." a sentence like "I am from India and I am" gets converted into something like [12, 24, 45, 11, 19, 10, 2], where each word gets its own number. these mappings aren't special - they're just like assigning id numbers to each word in a vocabulary.

but why do this? because computers are good with numbers, not text. they can do math with numbers, but they can't directly do math with the word "India." so we convert everything to numbers, do all our fancy math operations, and then convert back to words at the end.

ok so as we have numbers now, we can go ahead and actually understand how transformers really work.

## attention baby!

alright so here's where things get interesting. the big breakthrough with transformers was basically saying: fuck sequential processing, what if we just look at everything at once?

**quick detour: what are encoders and decoders?**

in the original transformer paper, they had this encoder-decoder setup for translation. but what does it really mean?! well ok think of two boxes that are connected together - each with a completely different job. the encoder box takes your input sentence (say, english) and processes the whole thing at once to build up this rich understanding of what it means. it's like someone reading and really getting what you wrote - the full context, the relationships between words, everything. and then it passes this deep understanding to the second box.

now the decoder box has the tougher job guys - it generates the output sentence (say, french) one word at a time, using both what it's already generated AND constantly checking back with the encoder's understanding. it's like having a translator who first completely understands your english sentence, then carefully crafts the perfect french translation word by word.

so encoder = understanding everything at once, decoder = generating piece by piece while referring back. let's say if you're translating something like "the cat sat on the mat," the encoder processes the whole english sentence in one shot, figuring out that there's a cat, it's sitting, and it's on a specific mat. then the decoder produces "le chat s'est assis sur le tapis" one word after another, checking back with the encoder's understanding to make sure it's capturing everything correctly.

and before we dive into encoder vs decoder drama, here's the thing - they're basically the same. like, the core building block is the attention mechanism, and both encoder and decoder use it. the only real difference is that decoders have this "causal masking" thing where each word can only look at previous words (because you're generating text left to right), and this is something called "autoregressive nature" while encoders let you look at the full sentence in both directions. but the actual attention computation? identical. so instead of explaining them separately and repeating myself, let's just understand how attention works, and then the encoder/decoder distinction becomes obvious.. let's goo.

### the core idea (why attention is actually genius)

so imagine you're translating "the cat sat on the mat" to french. when you're working on translating "sat", you probably want to look at "cat" (who's doing the sitting?) and "mat" (where are they sitting?). you don't really care about "the" at that moment.

that's attention. instead of forcing the network to remember everything in some compressed hidden state, you just let it look back at all the words and decide which ones matter right now. the network learns what to pay attention to.

here's what made it click for me, just hear me out: rnns are like someone telling you a story, and you're trying to remember everything they said. by the end, you've forgotten half of it. attention is like having the whole story written down in front of you, and you can just look at whichever part you need at any moment. at any time, you can just look back and observe the whole context again to make sense of the current moment.

okay but wait - if we're looking at all words simultaneously, how does the model know that "cat sat on mat" is different from "mat sat on cat"? word order obviously matters, right?

this is where positional embeddings come in.

### positional embeddings : gives you the wings

![Positional Embeddings](/videos/attention-is-all-you-need-guys/PositionalEmbeddings.gif)

before anything else happens, we take each word and turn it into a vector (word embedding - pretty standard stuff). but then we add positional information directly to these embeddings. position 1 gets one pattern added, position 2 gets another, and so on. imagine it as something which is just basic as telling the model where a specific word really is, that's what positional embeddings do.

the transformer uses these specific formulas to create position patterns:

• PE(pos, 2i) = sin(pos / 10000^(2i/d))

• PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

let me break it down so that it doesn't look freaky. pos is the position number in the sequence (1st word, 2nd word, etc). i is the dimension index we're calculating (0, 1, 2... up to d/2-1) as each i value creates two dimension values, 2i through sin and 2i+1 through cos. d is the total embedding size which is 512 for the original transformer. if d is 512, then i only needs to go from 0 to 255:

• i=0 gives dimensions 0 and 1

• i=1 gives dimensions 2 and 3

• ...

• i=255 gives dimensions 510 and 511

let me calculate actual values for different positions with d=512. say we have a sequence "the cat sat on mat", now position 1 is the word "the", at position 2 we have "cat":

for position 1 (pos=1):

• for dimension i=0:
  
  • PE(1, 0) = sin(1/10000^(0/512)) = sin(1/1) = sin(1) = 0.84

  • PE(1, 1) = cos(1/10000^(0/512)) = cos(1/1) = cos(1) = 0.54

• for dimension i=1:
  
  • PE(1, 2) = sin(1/10000^(2/512)) = sin(1/1.0027) = sin(0.997) = 0.84
  
  • PE(1, 3) = cos(1/10000^(2/512)) = cos(1/1.0027) = cos(0.997) = 0.55

for position 2 (pos=2):

• for dimension i=0:
  
  • PE(2, 0) = sin(2/10000^(0/512)) = sin(2/1) = sin(2) = 0.91
  
  • PE(2, 1) = cos(2/10000^(0/512)) = cos(2/1) = cos(2) = -0.42

• for dimension i=1:
  
  • PE(2, 2) = sin(2/10000^(2/512)) = sin(2/1.0027) = sin(1.994) = 0.91
  
  • PE(2, 3) = cos(2/10000^(2/512)) = cos(2/1.0027) = cos(1.994) = -0.41

as you can see, each position gets a completely unique vector of 512 values. position 1's vector starts with [0.84, 0.54, 0.84, 0.55...] while position 2's starts with [0.91, -0.42, 0.91, -0.41...].

there's one important thing here - the position embedding for position 1 is always exactly the same, no matter if it's the 1st word in "the cat sat on mat" or the 1st word in "transformers are awesome" or any other sequence. it's just telling the model, "hey, this word is at position 1, this word is at position 2" and so on.

ok so now we have two different embeddings for each word:
- the word embedding from our embedding model (what the word means)
- the positional embedding we just calculated (where the word is)

what do we do with them? super simple - we just add them together! that's literally it.

for each word, we take its word embedding vector, add the positional embedding vector for its position, and that combined vector becomes our input to the attention layers.

don't ask me why adding works... it just does. we're basically smashing together "what" and "where" information and trusting that during training, the model will figure out how to use both signals. seems kinda weird that simple addition works here, but the math checks out and the results speak for themselves. i mean this is the "attention" paper we are talking about.

during training, the model learns to interpret these combined vectors and somehow extract both the meaning of words and their positions when needed. it's one of those "surprisingly simple but effective" tricks in deep learning that just works!

### how it actually works (the full picture)

alright now let's go through the whole attention mechanism step by step. fasten your seat belt guys, because there's more to it than just "queries, keys, values".

before getting into this, I just want to clarify - the original attention paper used 512 as the size of the embedding passed into the model, and there are 8 heads with each head size being 64. there are some other defaults too, but for now just keep these in mind and we're good to go.

another thing - when I say word, I mean unigram token. there are certain architectures where the tokens can be bigram or maybe n-gram (depends on the architecture we use).

**step 0: starting point**

you have your input - let's say "cat sat mat" (3 words). each word gets converted to an embedding vector (let's say dimension 512), and positional encodings are added. so now you have three vectors, let's call them x₁, x₂, x₃, each of size 512.

**step 1: creating queries, keys, values through projection**

![QKV Mechanism](/videos/attention-is-all-you-need-guys/QKVMechanism.gif)

here's the thing people often skip - Q, K, V aren't just magically there. you CREATE them by multiplying your input by learned weight matrices. so basically you have three matrices:
W_Q, W_K, W_V (each of size 512 × 64)

for each position i:

- q_i = x_i × W_Q (this is your query)
- k_i = x_i × W_K (this is your key)
- v_i = x_i × W_V (this is your value)

these matrices are learned during training as different tasks need different kinds of attention, so the model learns what transformations produce useful queries, keys, and values through learning those weights.

**step 2: compute attention scores (scaled dot product)**

![Attention Scores](/videos/attention-is-all-you-need-guys/AttentionScores.gif)

okay quick recap of where we are: we started with word embeddings, added positional encodings, and now we have our input matrix. each word is a vector. then we projected these through weight matrices (W_Q, W_K, W_V) to create queries, keys, and values for each token.

now here's the thing - when we talk about multi-head attention (which we'll get to in detail soon), each head works with a smaller dimension. so if your full embedding is 512 dimensions and you have 8 heads, each head gets 512/8 = 64 dimensions. that's what d_k represents - the dimension per head. for now we're basically zooming in on one attention head. we'll come to multi head later, but for now, just consider there is one head with 64 different features it can see for a given sequence.

so let's say we're focusing on position 2 (word "sat") in one of these heads.

its query is q₂ (a 64-dimensional vector). we compute dot products with all keys:

```
score₁ = q₂ · k₁ (how much should "sat" attend to "cat")
score₂ = q₂ · k₂ (how much should "sat" attend to itself)
score₃ = q₂ · k₃ (how much should "sat" attend to "mat")
```

and now you get some scores.

these scores give an impression of how much one word is important to another in terms of attention. higher score = more relevant. but here's the critical part which often doesn't get enough recognition - we scale these scores by dividing by √d_k, where d_k is the dimension of the keys (64 in our example, since we're looking at one head).

![Scaled Dot Product](/videos/attention-is-all-you-need-guys/ScaledDotProduct.gif)

why? because without scaling, when d_k is large (think about the big architectures which have more dimensions per head), the dot products get really big. think about it - you're adding up 64 products of numbers. the more numbers you add, the bigger the sum gets on average. and when you put really big numbers into softmax, it becomes super peaky - basically all attention goes to one position (like 0.99, 0.005, 0.005). in that case, it's like saying only this position matters in the whole sequence, which is kinda wrong right? that's where dividing by √d_k keeps the variance stable and the softmax reasonable. this is a bit tricky at first glance but when you think about it for some time, you will actually give more respect to this small operation. i [posted more about this on my x](https://x.com/viplismism/status/2003807608571076782?s=20), would be helpful if you want to dig a bit deeper.

ok so now the actual scores are:

- score₁ = (q₂ · k₁) / √64 = (q₂ · k₁) / 8
- score₂ = (q₂ · k₂) / 8
- score₃ = (q₂ · k₃) / 8

this scaled dot-product attention is the core mechanism. simple but super effective.

**step 3: softmax to get attention weights**

now we run these scaled scores through softmax:

```
attention_weights = softmax([score₁, score₂, score₃])
```

what softmax does is turn any set of numbers into a probability distribution - all the scores after passed through softmax will sum to 1. so if our scaled scores were [1.2, 0.5, 1.1] for q2 which was "sat", then softmax might give us [0.42, 0.16, 0.42]. this means "sat" should pay 42% attention to "cat", 16% to itself, and 42% to "mat". but just to put a note here - [softmax is kinda dumb](https://x.com/viplismism/status/2004222322778296459?s=20), guys!

**step 4: weighted sum of values**

alright, now comes the payoff time. we have attention weights that tell us how much to focus on each position. and we have value vectors from each position that contain the actual information to pass along.

think about it this way - the query and key vectors were just for figuring out relevance. query says "what am i looking for?" and key says "what kind of info do i represent?". they just compute the attention scores. but the value vector? that's the actual content. that's "here's what i can actually give you".

like if you're searching for "best pizza in brooklyn", the query is your search, the keys are like those titles/descriptions that help you find relevant results, but the values are the actual articles/reviews you end up reading. the keys helped you find what's relevant, but the values are the real information you take away.

now combining this, we compute the output as a weighted average of these value vectors using the attention weights we got earlier:

```
output₂ = 0.42 × v₁ + 0.16 × v₂ + 0.42 × v₃
```

what's happening here? well to be honest we're just mixing information from different positions based on what we calculated earlier. it's like "sat" is grabbing 42% of the information from "cat", 42% from "mat", and only 16% from itself.

this output₂ is the new representation for position 2 after attention! look closely at how it started as just the word "sat", but now it's enriched with contextual information from "cat" and "mat". that's the magic - each word's representation gets updated based on relevant context.

now just imagine this whole process happening for all positions at the same time. yeah, position 1's query is attending to all keys and getting its weighted values, position 2 is doing the same, position 100 is doing it - everyone in parallel. each position gets a new contextual representation based on how much that word should attend to all the other words. just read it once more! this is literally it, the whole gist of attention.

this is huge for speed btw. remember how rnns had to process word 1, then word 2, then word 3 sequentially? well here, every position computes its attention simultaneously. you can throw this at a gpu and it just parallelizes everything. that's why transformers are so much faster too.

**the full picture in one formula:**

time to see everything at once:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

now as i said earlier, imagine this attention thing happening for all positions at the same time. these are just matrices after all. Q is all your queries stacked as rows (one row per position), K is all your keys stacked as rows, V is all your values stacked as rows.

when you do QK^T, you're computing every query with every key in one shot. it's a matrix multiplication that gives you a grid of scores - position 1's query against all keys, position 2's query against all keys, and so on. you scale everything by √d_k, then apply softmax row-wise (each row becomes probabilities that sum to 1), then multiply by V to get all the weighted sums at once.

boom, you've computed attention for all positions in parallel. no sequential processing, just big matrix multiplications that gpus are really good at. you can process a 1000-word sequence and every position attends to every other position simultaneously. how cool is that!

### multi-head attention (because one perspective isn't enough)

![Multi-Head Attention](/videos/attention-is-all-you-need-guys/MultiHeadAttention.gif)

okay so here's where it gets even cooler. what if we had multiple attention mechanisms running in parallel, each learning to focus on different aspects? remember earlier I said something about the number of heads?

it's like understanding a sequence through different lenses. one head might capture grammatical relationships between words (like subject-verb agreement), another head might be better at understanding semantic connections (like "bank" relating to "river" vs "bank" relating to "money"), another might focus on long-range dependencies, and yet another might specialize in local context.

just think about how you read a sentence yourself. you're simultaneously tracking multiple things - sometimes it's grammar or meaning, sometimes it's references to earlier parts, tone, relationships between entities. you're not just looking at it one way. multi-head attention does the same thing - it processes the sequence from multiple angles at once.

that's multi-head attention. instead of one set of W_Q, W_K, W_V matrices, you have h sets (the paper uses h=8). so you have:

- head 1: W_Q¹, W_K¹, W_V¹ → produces attention output₁
- head 2: W_Q², W_K², W_V² → produces attention output₂
- ...
- head 8: W_Q⁸, W_K⁸, W_V⁸ → produces attention output₈

each head is smaller (if total dimension is 512 and you have 8 heads, each head works with 64 dimensions). they all run the same attention mechanism, just with different learned projections.

just don't confuse here - whatever we understood earlier about how attention works? that's exactly what's going on in all the other heads too. same queries, keys, values, same dot products, same softmax, same weighted sums. the computation is identical.

the only difference is instead of working with the whole 512-dimensional space, each head works on a smaller slice. if you have 8 heads, each one gets 64 dimensions (512/8). so head 1 might be working with dimensions 0-63, head 2 with dimensions 64-127, and so on. same attention mechanism, just on different parts of the representation.

and here's the cool part - even though they're all doing the same computation, they learn to specialize during training. maybe head 1 learns to focus on subject-verb relationships, head 2 picks up on spatial/location words, head 3 captures temporal markers like "yesterday" or "will". the model figures out these specializations on its own just from the training objective. you don't tell it what to look for - it discovers useful patterns.

**combining the heads**

after you've computed all 8 attention outputs, you concatenate them:

```
concat = [output₁, output₂, ..., output₈]
```

but you can't just leave it like that - you need to mix the information from different heads together. so there's one more learned matrix W_O that you multiply the concatenated heads by:

```
final_output = concat × W_O
```

this W_O matrix is really crucial guys - it lets the model learn how to combine the different perspectives from each head into one coherent representation. without it, the heads would just be sitting next to each other not really talking to each other. it's like having different juices in different containers, but we want to drink a mix of them. [that's what W_O is for](https://x.com/viplismism/status/2009587974846173656?s=20).

### feed-forward network (the other half of the layer)

alright so attention is cool and all, but there's another component in each transformer layer that often gets less spotlight - the feed-forward network.

but before we get there, we need to talk about two things that happen between attention and ffn: residual connections and layer normalization. these might sound boring but they're actually crucial for making deep networks trainable.

**residual connections (aka skip connections)**

![Residual Connections](/videos/attention-is-all-you-need-guys/ResidualConnections.gif)

okay so remember the vanishing gradient problem from rnns? turns out when you stack many layers of any neural network, you can run into similar issues. when i say many layers here, i'm talking about the entire transformer block (attention + ffn and other linear projections) getting repeated multiple times. (more on ffn in a bit) the original paper stacks 6 of these blocks, GPT-3 has 96, and we'll get into why and how this stacking works later. but for now, just know that if you have many of these blocks stacked on top of each other, during backprop the gradient gets weaker and weaker as it travels backwards.

because when you backpropagate, you're using the chain rule. the gradient at block 1 depends on the gradient at block 2, which depends on block 3, and so on. so you end up multiplying a bunch of gradients together, one from each block. and if most of these multiplications involve numbers less than 1 (which they often do), you're multiplying 0.8 × 0.7 × 0.9 × ... and after a couple of blocks, that product gets really tiny.

it's like the telephone game, but with math. each block adds a bit of "loss" to the signal. by the time the gradient reaches the early blocks, it's so weak that those layers barely learn anything as the weights don't update meaningfully because the gradient is basically zero.

well the fix is surprisingly simple: instead of just passing the attention output to the next step, you add it to the original input. so if x was your input to the attention layer, and attention(x) is the output, you actually use x + attention(x). that's it, literally.

residual connections are like emergency exits for your neural network. when information passes through attention layers, some of it might get lost or distorted because there's just too much dependency on layers, one after another.

but with residual connections, we basically say "hey, let's also keep a copy of the original input and just add it back after the transformation." this creates a direct highway for both information and gradients to flow through the network. so during training, even if the gradient signal gets weak going through the attention block, it can still travel through this shortcut path without any disturbance. that's why we can stack these blocks 12+ layers deep without the network becoming untrainable.

and btw the math behind residual connections is beautifully simple (i know it's too much but just stick with it...)

```
y = x + F(x)
```

where:
- x is your input (like the word embeddings + positional embeddings)
- F(x) is the transformation (attention layer)
- y is your output

during backpropagation, we calculate ∂L/∂x - the gradient of loss with respect to input:

```
∂L/∂x = ∂L/∂y × (1 + ∂F(x)/∂x)
```

this is where the magic happens - that "+1" term means even if ∂F(x)/∂x becomes tiny (like 0.0001) through many layers, we still have that constant 1 guaranteeing gradient flow.

with standard networks (y = F(x)), your gradient is just ∂L/∂y × ∂F(x)/∂x, which can shrink to almost nothing after many layers. but that +1 from residual connections creates an express lane for gradients that never gets blocked, no matter how deep you go.

**layer normalization**

![Layer Normalization](/videos/attention-is-all-you-need-guys/LayerNormalization.gif)

after each sub-layer (attention or feed-forward), we apply something called layer normalization.

[but what exactly is it doing?](https://x.com/viplismism/status/2000517100071420358?s=20)

here's a simple way to think about layer normalization.

imagine you're trying to teach a group of students. some are really loud talkers and others whisper. the loud ones dominate every conversation while the quiet ones get ignored. that's basically what happens in neural networks without normalization.

in a neural network, different neurons can have wildly different "volumes" - some might output huge numbers like 1000, others tiny ones like 0.01. when this happens, the big numbers bulldoze through the network and the small ones get lost in the noise. it's like having one person shouting in a room full of whispers. that's a problem as this basically nullifies the relevance of the quieter neurons.

layer normalization is like giving everyone the same volume control. it takes all the outputs from a layer and says "hey, let's make sure you're all speaking at roughly the same level." it doesn't change what each neuron is trying to say, just normalizes how loudly they're saying it.

why does this matter?! well, when all neurons are "speaking" at similar volumes, the network can actually learn from everyone. the gradients flow better during training, convergence happens faster, and the whole system becomes more stable. it's like finally being able to hear that quiet person in the back who actually had great ideas all along...

without normalization, training can be painfully slow or just break entirely. with it, your network learns more efficiently and generalizes better. pretty neat for something so conceptually simple, right?

in our case, layer norm looks at each position's vector (512 dimensions) and normalizes it across the feature dimension. it's like saying "let's make sure no single position has wildly different activation patterns than the others."

mathematically, for each position's vector, we:

- calculate the mean (μ) and standard deviation (σ) across all feature dimensions (all 512 columns)
- subtract the mean from each value: (x - μ)
- divide by standard deviation: (x - μ)/σ
- apply learnable scale (γ) and shift (β) parameters: γ × ((x - μ)/σ) + β

this is it. doing this makes sure the mean is 0 and the variance is 1 across the feature dimension.

in transformers specifically, layer norm is critical because attention weights can cause values to grow unpredictably. normalization ensures that even after 12+ layers, the activations don't explode or vanish.

**now to the feed-forward network**

![Feed Forward Network](/videos/attention-is-all-you-need-guys/FeedForwardNetwork.gif)

alright, so after multi-head attention → residual connection → layer norm, you've got a normalized output which is still a matrix of shape (sequence_length × 512).

now this goes into the feed-forward network. and here's the key thing - the ffn processes each position independently. it's applied to each row of this matrix separately. no mixing across positions here, just transforming each position's representation on its own.

the structure is super simple:

```
ffn(x) = max(0, xW₁ + b₁)W₂ + b₂
```

let's break down what's actually happening with the dimensions.

- input to ffn: one position's vector, size 512
- W₁: weight matrix of size 512 × 2048 (expands to 4x larger)
- after W₁ + bias + ReLU: you get a 2048-dimensional vector
- W₂: weight matrix of size 2048 × 512 (compresses back down)
- output: back to 512 dimensions

so you're expanding each position's representation to a much higher dimension (2048), applying non-linearity with ReLU, then compressing back to the original size. this happens identically for every position in the sequence. expanding to higher dimensions is like giving the model a bigger "workspace" to compute complex transformations that wouldn't be possible in the original 512-dim space.

the higher dimensional space lets the network learn really cool patterns and relationships - like detecting subtle linguistic features, syntactic structures, or semantic associations that require multiple intermediate computations. it's kind of like how you might need scratch paper to solve a complex math problem, right?! the 2048 dimensions give you that scratch space, then compress back to 512 so it can be passed to later layers for the next set of computations.

and yeah, W₁ and W₂ are learned during training just like the attention matrices. in fact, most of the parameters in a transformer are actually in these feed-forward layers, not the attention! if you count parameters, the FFN is usually where most of them live.

**the full layer structure**

so putting it all together, one transformer layer looks like:

1. multi-head attention (mix information across positions)
2. add & norm (residual connection + layer normalization)
3. feed-forward network (process each position independently)
4. add & norm again (another residual + layer norm)

then you stack like 6 or 12 or 96 of these layers depending on how big you want your model. more layers means more parameters to train, and each layer refines the representations further. the first layer might capture basic patterns, middle layers get more abstract relationships, and later layers handle really high-level understanding.

that's it, that's your attention thing!

### the autoregressive thing (why decoders are slightly different)

![Causal Masking](/videos/attention-is-all-you-need-guys/CausalMasking.gif)

okay so remember i said encoders and decoders are basically the same? here's the one difference: when you're generating text (like in GPT or any modern LLM), you can't look at future words because they don't exist yet. you're predicting the next word based only on previous words.

so in decoder attention, when you're at position 3 (say, the word "sat"), you can look at positions 1 and 2 ("the" and "cat") and yourself (position 3), but not at positions 4 and 5 ("on" and "mat") because those haven't been generated yet.

you implement this with a mask. before the softmax, you set the attention scores for all future positions to negative infinity. after softmax, these become exactly zero. so those positions contribute nothing to the weighted sum.

same attention mechanism, just with this causal mask. this is why GPT-style models only use decoders - for text generation, you only need the causal version anyway.

encoders (like in BERT) let you look at the whole sentence in both directions, which is great for understanding tasks like "is this review positive or negative?" but can't generate text word by word. decoders are one-directional, perfect for generation.

and honestly? most modern LLMs just use decoders now. turns out you don't need the encoder for most things. the decoder-only architecture is simpler and scales better.

### a quick note on kv-cache (the inference trick)

now here's something interesting that becomes super important when you actually run these models in production. when generating text token by token, you'd normally have to recompute the keys and values for all previous tokens at every step. think about it - if you've generated 500 tokens and want to predict the 501st, you'd recompute K and V for all 500 tokens again. that's wasteful as hell, right?

kv-cache is the simple but brilliant fix. since the keys and values for already generated tokens don't change (they only depend on the input, not future tokens), you just... cache them. store them in memory. so when generating token 501, you only compute the new query for position 501, use the cached keys and values from positions 1-500, and compute just one new key-value pair to add to the cache.

this makes inference way faster, especially for long sequences. the tradeoff is memory - you're storing these cached tensors which can get pretty large for long contexts. there's a whole rabbit hole here about different caching strategies, memory optimization, and how this interacts with things like flash attention. but that's a story for another blog. more on kv-cache and inference optimizations coming soon!

so that's it. if you have come along this far reading everything, trust me, there are close to 6000 words in this writeup with almost 34k characters. i hope this actually made you understand everything about attention. until next time, adios!