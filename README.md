## Countdown Prompt Priming

This repo contains code to reproduce experiments from the blogpost Priming language models with cognitive behaviors via prompting for better RLVR: 

It supports:

- zero-shot training
- one-shot training with the exact Countdown example from the post
- checkpoint evaluation under either zero-shot or one-shot prompting
- exporting responses so you can inspect traces and compute sentence lengths

## Setup

```bash
uv sync
```

Set `TINKER_API_KEY` in your environment before running train or eval.

## Reproduce The Main Runs

Train the zero-shot run:

```bash
uv run python -m experiments.countdown.rl.train run_name=llama_countdown_zero_shot prompt_style=zero_shot
```

Train the one-shot run:

```bash
uv run python -m experiments.countdown.rl.train run_name=llama_countdown_one_shot prompt_style=few_shot
```

Defaults are aligned to the blogpost setup:

- `model_name=meta-llama/Llama-3.1-8B-Instruct`
- `group_size=8`
- `groups_per_batch=64`
- `train_size=8192`, which gives `16384 / 64 = 256` GRPO steps
- `save_every=32`

## Reproduce The Three Evaluation Settings

1. One-shot at train-time, one-shot at test-time:

```bash
uv run python -m experiments.countdown.rl.inference \
  checkpoint_dir=runs/countdown/rl/llama_countdown_one_shot \
  prompt_style=few_shot \
  output_file=outputs/one_shot_train_one_shot_test.json
```

2. One-shot at train-time, zero-shot at test-time:

```bash
uv run python -m experiments.countdown.rl.inference \
  checkpoint_dir=runs/countdown/rl/llama_countdown_one_shot \
  prompt_style=zero_shot \
  output_file=outputs/one_shot_train_zero_shot_test.json
```

3. Zero-shot at train-time, zero-shot at test-time:

```bash
uv run python -m experiments.countdown.rl.inference \
  checkpoint_dir=runs/countdown/rl/llama_countdown_zero_shot \
  prompt_style=zero_shot \
  output_file=outputs/zero_shot_train_zero_shot_test.json
```

The inference default is `n_problems=50`.

## Checkpoints And Trace Lengths

Evaluate a saved checkpoint such as step 64:

```bash
uv run python -m experiments.countdown.rl.inference \
  checkpoint_dir=runs/countdown/rl/llama_countdown_one_shot \
  step=64 \
  prompt_style=few_shot \
  output_file=outputs/step64_one_shot.json
```

Summarize accuracy and mean sentence count from an exported inference file:

```bash
uv run python -m experiments.countdown.rl.analyze outputs/step64_one_shot.json
```
