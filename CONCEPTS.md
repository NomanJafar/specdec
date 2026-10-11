# Concepts: speculative decoding, or how to get several tokens for the price of one

`specdec` runs vLLM with a small draft model proposing tokens for a larger target model, tracks how many proposals are accepted, and measures what that does to latency and throughput at different batch sizes. This primer explains why it can be faster at all, why it is exactly lossless, and when it stops helping.

## 1. The waste speculative decoding exploits

Decode is memory-bound (servellm primer, section 1): each step streams all the weights through the GPU to produce one token per sequence, and at small batch sizes the arithmetic units sit mostly idle. If we could somehow present the model with *k* candidate next tokens at once, it could score all of them in a single forward pass for almost the same cost as scoring one, because the extra work is arithmetic, which is nearly free, not weight traffic, which is the bottleneck.

The problem is that we do not know the next k tokens; that is the whole point of decoding. Speculative decoding's move is to *guess* them cheaply and *verify* them in one pass.

## 2. Draft, verify, accept

One speculative step:

1. **Draft.** A small, fast model (the draft; here Qwen2.5-0.5B) generates k tokens autoregressively: k cheap forward passes.
2. **Verify.** The large model (the target; here Qwen2.5-3B) runs **one** forward pass over the prompt plus the k drafted tokens and produces its own next-token distribution at every position (it would have produced those anyway; a transformer computes logits for all positions in the input).
3. **Accept.** Walk the draft left to right. At each position, compare the draft's token against the target's distribution with the acceptance rule below. The first rejection ends the walk; everything before it is kept, the rejected position gets a token sampled from an adjusted target distribution, and the draft tokens after it are discarded.
4. If all k are accepted, the target's distribution at position k+1 gives one more token for free.

So one target forward pass yields between 1 and k+1 tokens. The expected number is the speedup lever.

## 3. Why it is lossless

The acceptance rule (Leviathan et al., Chen et al., 2023) is rejection sampling in disguise. For draft distribution q and target distribution p at a position with drafted token x:

- accept x with probability `min(1, p(x) / q(x))`;
- on rejection, sample a replacement from `norm(max(0, p − q))`.

The tokens produced by this procedure are distributed **exactly** as if sampled from p alone. Not approximately; the proof is a few lines of algebra. With greedy decoding (temperature 0) it reduces to "accept if the draft token equals the target's argmax", and the output is identical token for token to running the target alone.

This is the part that makes speculative decoding unusual among inference optimisations: quantization, pruning and distillation all trade quality for speed; this trades *compute* for speed and the output distribution does not move. The "free lunch" in the project title is this property. The cost is paid in wasted work (rejected drafts) and in memory (two models, two KV caches).

## 4. Acceptance rate is everything

Let α be the per-token acceptance probability (assume it is roughly constant). The expected number of tokens from one verify step with k drafts is

```
E[tokens] = (1 − α^(k+1)) / (1 − α)
```

| α | k=1 | k=3 | k=5 |
|---|---|---|---|
| 0.5 | 1.5 | 1.9 | 2.0 |
| 0.7 | 1.7 | 2.5 | 3.0 |
| 0.9 | 1.9 | 3.4 | 4.7 |

The step costs one target pass plus k draft passes plus the verification overhead. The wall-clock speedup is

```
speedup ≈ E[tokens] / (1 + k × c_draft + c_overhead)
```

where `c_draft` is the draft's step time as a fraction of the target's. A 0.5B draft against a 3B target is roughly c ≈ 0.2 (bandwidth-bound, proportional to weight size). With α = 0.7 and k = 3: 2.5 / (1 + 0.6) ≈ 1.6x. With α = 0.5: 1.9 / 1.6 ≈ 1.2x. With α = 0.3 it is a slowdown. **Acceptance rate is the number to watch**, and vLLM exports it (`vllm:spec_decode_num_accepted_tokens_total` over `vllm:spec_decode_num_draft_tokens_total`, plus a per-position acceptance histogram in newer versions).

What drives α: how similar the draft's predictions are to the target's. Same family and tokenizer (Qwen 0.5B → 3B) helps; predictable text (code, boilerplate, repeated formats) gives high α; creative or high-entropy text gives low α; higher sampling temperature lowers α because the target is more spread out.

## 5. Choosing k

More drafts per step give more upside when α is high and more waste when α is low: with α = 0.5, going from k = 3 to k = 5 buys 0.1 extra tokens at the cost of two more draft passes. The sweep over `num_speculative_tokens` 1–5 in this project is designed to find the knee for this pair and this text. Expect it around 3–4 for α near 0.7, lower for harder text.

## 6. Where it stops helping: batch size

The whole premise is idle arithmetic units at small batch. As the batch grows, decode becomes compute-bound (servellm results: ITL rising from concurrency 8 upward on the T4). Verification of k drafted tokens for B sequences costs the same arithmetic as a batch of B × (k+1) ordinary tokens; once that is compute-bound, the verify pass is no longer "almost free", and the rejected drafts are pure waste. Speculative decoding then reduces throughput. Production systems turn it off above some batch size or make k adaptive (vLLM has a `disable_by_batch_size`-style option). The sweep at concurrency 1, 4 and 16 is there to show the crossover on this hardware.

## 7. Memory

Two models and two KV caches share the GPU. On a 15 GiB T4 with the 3B target in FP16 (~6.2 GiB) and a 0.5B draft (~1 GiB), the KV cache budget drops from ~8 to ~7 GiB, and the draft's own cache takes a slice of that. kvcalc's arithmetic applies to each model separately. For large batches this memory cost compounds the compute argument against speculation.

## 8. Drafts without a draft model

**n-gram (prompt lookup) speculation** proposes the tokens that followed the most recent n-gram the last time it appeared in the prompt or the generated text. No second model, no extra memory, essentially free drafts; α is high when the output repeats the input (summarisation, extraction, code editing, RAG answers that quote the context) and near zero otherwise. vLLM supports it as `method: ngram`. This project measures it as a second method on the same prompts so the two can be compared on the same axis: acceptance rate.

Other draft sources, not measured here: **Medusa** and **EAGLE** heads trained onto the target model itself (higher α, no separate model, but training required), and **self-speculation** by skipping layers.

## 9. What the experiment will show

1. Baseline: target alone, concurrency 1, 4, 16: ITL and throughput.
2. Draft model speculation with `num_speculative_tokens` 1, 3, 5 at the same concurrencies: acceptance rate from `/metrics`, ITL, throughput, and the realised speedup against the baseline.
3. n-gram speculation on the same prompts, plus on a prompt set designed to repeat its input (summarise-this-paragraph), to show α moving with the text.
4. The crossover: the concurrency at which speculation stops improving throughput on the T4.
5. A spot check that greedy outputs are byte-identical with and without speculation (the losslessness claim, verified rather than asserted).

## Further reading

- Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023).
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023): the acceptance rule and proof.
- Saxena, *Prompt Lookup Decoding* (2023): n-gram speculation.
- Cai et al., *Medusa* (2024); Li et al., *EAGLE* (2024): drafts from trained heads.
- vLLM docs: *Speculative Decoding* page for the current configuration keys and metrics.
