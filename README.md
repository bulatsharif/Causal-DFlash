# Causal-DFlash

**Target-conditioned causal diffusion drafter for speculative decoding**

Causal-DFlash is a research project exploring whether the parallel drafting mechanism of
[DFlash](https://arxiv.org/abs/2602.06036) can be combined with the cache-friendly causal
diffusion architecture introduced in [WeDLM](https://arxiv.org/abs/2512.22737).

The main goal is to build a lightweight diffusion drafter that predicts several future tokens
in parallel while remaining compatible with standard causal attention, KV caching, and
optimized autoregressive inference engines.

> Research project for **Summer with AIRI 2026 — Efficient Deep Learning track**.

---

## Motivation

Speculative decoding accelerates LLM inference by using a small draft model to propose several
future tokens, which are then verified in parallel by the target LLM.

DFlash improves drafting latency by replacing an autoregressive drafter with a lightweight block
diffusion model. However, its block-wise bidirectional attention is not naturally compatible with
standard prefix KV caching, and its practical speedup may vary substantially across tasks and
serving configurations.

WeDLM demonstrates that diffusion-style generation can operate under standard causal attention
through topological reordering. This makes predicted tokens immediately cache-compatible and
allows the model to use existing inference infrastructure such as FlashAttention, PagedAttention,
and CUDA Graphs.

Causal-DFlash investigates whether these ideas can be transferred to a small target-conditioned
speculative drafter.

---

## Proposed Architecture

<p align="center">
  <img src="assets/causal_dflash_architecture.png"
       alt="Causal-DFlash architecture"
       width="900">
</p>

The proposed pipeline consists of four stages:

1. **Target context extraction**

   The target LLM processes the current prefix. Hidden states from several target-model layers
   are fused into compact context features.

2. **Causal diffusion drafting**

   A lightweight drafter receives the current prefix followed by a fixed window of masked
   positions. Target context features are injected into every draft layer through KV projections.

3. **Multi-step draft generation**

   Topological reordering places currently observed tokens before unresolved masks while
   preserving their logical position IDs. The drafter progressively resolves the complete draft
   block over several internal diffusion steps.

   For example, an 8-token block may be generated as:

   ```text
   Step 1: +3 tokens  → 3 / 8 resolved
   Step 2: +2 tokens  → 5 / 8 resolved
   Step 3: +3 tokens  → 8 / 8 resolved
