# specdec

Speculative decoding pipeline: small draft model plus large target, with per-request acceptance-rate tracking and measured speedup.

Project 6 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

A small draft model proposes tokens, a large target model verifies them; track acceptance rate and the resulting speedup per request.

## Why it matters

Faster decode with zero quality loss is the closest thing to a free lunch in inference.

## Status

Primer, analysis code and the notebook are done; the GPU run is next. Results land in `results/` and below once it has run.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/specdec/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
