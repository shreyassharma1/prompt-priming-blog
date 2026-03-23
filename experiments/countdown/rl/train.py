"""Train a model on Countdown puzzles with RL (GRPO)."""

import asyncio
import logging
from pathlib import Path
from typing import Literal

import chz
from reasoning_gym.games import CountdownConfig, CountdownDataset

from experiments.reasoning_gym.prompting import build_llama_convo_prefix
from experiments.reasoning_gym.rl_env import (
    countdown_format_ok,
    extract_countdown_answer,
)
from experiments.rl_common.core import configure_logging
from experiments.rl_common.reasoning_gym_runner import run_reasoning_gym_training

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs" / "countdown" / "rl"


@chz.chz
class TrainConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name: str | None = None
    lora_rank: int = 32

    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 1e-5
    max_tokens: int = 1024
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, object] | None = None
    num_substeps: int = 1
    temperature: float = 1.0
    compute_post_kl: bool = False
    remove_constant_reward_groups: bool = False

    run_name: str | None = None

    min_numbers: int = 3
    max_numbers: int = 4
    min_value: int = 1
    max_value: int = 100
    min_target: int = 100
    max_target: int = 999
    operators: tuple[str, ...] = ("+", "-", "*", "/")
    shuffle: bool = True

    train_size: int = 16384
    test_size: int = 100
    seed: int = 0

    reward_mode: str = "graded"
    format_coef: float = 0.0

    save_every: int = 32
    eval_every: int = 8

    prompt_style: Literal["zero_shot", "few_shot"] = "zero_shot"


def _dataset_factory(config: TrainConfig, size: int, seed: int) -> CountdownDataset:
    cfg = CountdownConfig(
        min_numbers=config.min_numbers,
        max_numbers=config.max_numbers,
        min_value=config.min_value,
        max_value=config.max_value,
        min_target=config.min_target,
        max_target=config.max_target,
        operators=config.operators,
        shuffle=config.shuffle,
        seed=seed,
        size=size,
    )
    cfg.validate()
    return CountdownDataset(cfg)


async def run_training(config: TrainConfig) -> str:
    return await run_reasoning_gym_training(
        config,
        runs_dir=RUNS_DIR,
        dataset_name="countdown",
        dataset_factory=_dataset_factory,
        answer_extractor=extract_countdown_answer,
        format_checker=countdown_format_ok,
        convo_prefix=build_llama_convo_prefix(config.model_name, config.prompt_style),
        require_extracted_answer_for_scoring=True,
        logger=logger,
    )


def main_cli() -> None:
    configure_logging()
    config = chz.entrypoint(TrainConfig)
    output_dir = asyncio.run(run_training(config))
    print("\nTraining complete!")
    print(f"Output directory: {output_dir}")
    print("\nTo run inference, use:")
    print(f"  python -m experiments.countdown.rl.inference checkpoint_dir={output_dir}")


if __name__ == "__main__":
    main_cli()
