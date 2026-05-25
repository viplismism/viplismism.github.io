---
layout: post
title: Running Terminal Bench 2.0 properly (and what 50+ hours of compute taught me)
description: notes from running terminal bench 2.0 end-to-end on GLM-4.7 — reproducing the leaderboard score, getting wrecked by concurrency-based rate limiting, and what the traces reveal about how agents actually fail.
tags: [AI Agents, Benchmarks, Evals, Terminal Bench]
version: Draft
release: 24-05-2026
---
spent the last week running terminal bench 2.0 properly and holy shit, the infrastructure complexity is way deeper than i expected. ended up burning through 50+ hours of compute time and learned some painful lessons about rate limiting, pass@k metrics, and what these benchmarks actually measure.

## so what is terminal bench actually?

well terminal bench isn't your typical "write a function to reverse a string" benchmark. it's this Stanford/Laude Institute framework that throws 89 real-world, end-to-end terminal tasks at your agent — stuff like compiling entire codebases with broken dependencies, training ML models from scratch with messy data, setting up production servers with conflicting configs, debugging system installations that fail silently.

the key difference from something like SWE-bench (which is github PR tasks) is that terminal bench tests if your agent can handle the full chaos of what developers actually do in terminals — dependency management, multi-step debugging, system configuration, the whole workflow. i mean it's just how the real developers work, all that fancy debugging and all.

these are complete tasks that experienced developers would take hours or days to solve, not isolated coding problems.

each task comes with three things: instruction describing what needs to be done, a containerized docker environment (isolated, reproducible), and verification tests that actually check if you solved it correctly. your agent gets thrown into the container and has to navigate the file system, chain multiple commands together, reason over long outputs, recover from failures, and complete the entire workflow autonomously, so it's like doing the things how a developer does.

## ok so how does this actually work in practice?

well i wanted to verify GLM-4.7's reported 41% score on the leaderboard, so i set up two runs: one with GLM-4.7 (hosted deployment) and one with GLM-4.7-official (straight from z.ai API). used harbor framework for orchestration and terminus-2 as the agent wrapper.

first attempt was on local machine with `CONCURRENT=8` (trying to be smart and parallelize everything). complete disaster. 12 out of 89 tasks failed immediately with `EnvironmentStartTimeoutError` before the agent even got to run — docker containers weren't spinning up properly, resource contention was brutal, the whole thing was chaos.

moved to GCP (n1-standard-8 VM) with proper docker support, and suddenly all 89 tasks started correctly. infrastructure matters way more than people think.

## so here's where it gets interesting with the results

- GLM-4.7-private (my deployment): **39.3%** (35/89 tasks)
- GLM-4.7-official (z.ai API): **39.3%** (35/89 tasks)
- official leaderboard score: **41%**

wait, why the 2 point gap? well turns out the leaderboard reports pass@5 (you get 5 attempts per task, if any succeeds you win), while my runs were pass@1 (single attempt per task). that ~2 point variance makes perfect sense — some tasks are flaky or the agent gets unlucky on first try.

the fact that private and official deployments hit identical 39.3% validates that the model itself is consistent. no API magic, no special sauce, just the model doing its thing.

## the rate limiting nightmare that made this take 50+ hours

initially thought z.ai would have standard RPM (requests per minute) limits like openai. nope. they use concurrency-based rate limiting — meaning they limit the number of simultaneous in-flight requests, not how many requests you make per minute.

started with `CONCURRENT=2`, immediately hit 429 errors. backed down to `CONCURRENT=1`, which means the agent processes exactly one task at a time, sequentially. 89 tasks × ~40 minutes average per task = 50+ hours of runtime.

this is brutally inefficient compared to openai's setup where you could have dozens of parallel evaluations as long as you stay under the per-minute token limits. concurrency-based limiting forces you into serial execution for anything that makes lots of requests.

## ok so what do the actual traces reveal about failure modes?

each task generates a full episodic trace: agent → terminal command → observation → agent reaction, repeated until completion or timeout. these traces are stored in directories like `train-fasttext__YAHHW6m/` with `agent/episode-0/` through `agent/episode-N/`, plus `trajectory.json` with the full conversation and a `verifier/` folder with the final reward (0 or 1).

looking at `train-fasttext` (a failed task): agent went through 175 episodes trying to train a fasttext model. started with missing dependencies, installed them, hit version conflicts, resolved those, then the training script had encoding issues, fixed that, then memory allocation failed, increased limits, then finally hit a data format problem it couldn't recover from after 40 minutes.

total tokens burned: 14M tokens across all episodes. at standard API pricing (~$3/M input, $15/M output), that's $8.64 for a single failed task. multiply that across 89 tasks with varying episode counts and you understand why benchmark evals are expensive as hell.

the traces expose three critical failure patterns that static benchmarks completely miss:

**dependency hell:** agents spend 20+ episodes just trying to install the right package versions. they install something, it breaks another dependency, they fix that, a third thing breaks. humans would give up and use a different approach. agents just keep looping until timeout.

**recovery loops:** agent fixes one thing, breaks another, fixes that, the first thing breaks again. no high-level replanning, just tactical fixes that create new problems. you see this especially on tasks involving system configuration where changing one setting has cascading effects.

**token explosion on failures:** successful tasks are efficient (5-10 episodes, maybe 100k tokens). failed tasks burn millions of tokens because the agent keeps trying variants that don't work. this is why pass@1 vs pass@5 matters economically — if you're doing 5 attempts per task, your token costs explode on the hard tasks that fail repeatedly.

## task-by-task: where private and official diverge

comparing task-by-task results between private and official deployments:

**easy tasks** (`cobol-modernization`, `qemu-startup`): both deployments passed these consistently. these are the tasks where there's a clear linear path and not many failure modes.

**hard tasks** (`train-fasttext`, `make-doom-for-mips`): both deployments failed these. these require either deep domain knowledge or perfect execution across 10+ dependent steps where one mistake cascades into failure.

**medium tasks:** this is where variance shows up. some tasks passed on private but failed on official (or vice versa) due to different random seeds, slight prompt variations, or just agent getting lucky/unlucky with its approach.

the leaderboard SOTA is openai codex CLI (gpt-5 powered) at 49.6%, with the base gpt-5-codex model hitting 42.8%. even the best frontier models can't crack 50% on this benchmark, and that's AFTER terminal bench 2.0 removed all the impossible/flaky tasks from v1.0.

## why this matters more than you think

terminal bench catches the exact failure modes that break production agents. it's not testing if your agent can write correct code in isolation — it's testing if your agent can actually complete real developer workflows that require context switches, recovery from failures, system-level reasoning, and chaining 5-10 commands with proper error handling.

if you're evaluating agents without terminal bench, you're basically optimizing for performance on toy problems while missing the chaos that happens when agents try to do real work in actual terminal environments.

the framework is fully open source ([github.com/laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench)), harbor makes it easy to run at scale across cloud providers, and the v2.0 release fixed all the verification issues from v1.0. if you're serious about agent evals, this is the one that actually matters.

also verified that GLM-4.7 hits exactly what they claimed (within pass@1 vs pass@5 variance), which is rare in this space where benchmarks often have reproduction issues.

peace out.
