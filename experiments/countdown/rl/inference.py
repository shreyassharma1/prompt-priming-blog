"""Run inference on a trained Countdown checkpoint."""

import asyncio
import logging
from typing import Literal

import chz
from reasoning_gym.games import CountdownConfig, CountdownDataset

from experiments.reasoning_gym.prompting import build_llama_convo_prefix
from experiments.reasoning_gym.rl_env import (
    countdown_format_ok,
    extract_countdown_answer,
)
from experiments.rl_common.core import configure_logging
from experiments.rl_common.reasoning_gym_runner import run_reasoning_gym_inference_cli

logger = logging.getLogger(__name__)


@chz.chz
class InferenceConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name: str | None = None
    checkpoint_dir: str
    step: int | None = None

    n_problems: int = 50
    seed: int = 42
    max_tokens: int = 1024
    temperature: float = 1.0
    n_samples: int = 1
    max_concurrency: int = 50
    prompt_style: Literal["zero_shot", "few_shot"] = "zero_shot"

    output_file: str | None = None

    min_numbers: int = 3
    max_numbers: int = 4
    min_value: int = 1
    max_value: int = 100
    min_target: int = 100
    max_target: int = 999
    operators: tuple[str, ...] = ("+", "-", "*", "/")
    shuffle: bool = True


def dataset_factory(config: InferenceConfig, size: int, seed: int) -> CountdownDataset:
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


async def main_async(config: InferenceConfig):
    return await run_reasoning_gym_inference_cli(
        config,
        dataset_name="Countdown",
        dataset_factory=dataset_factory,
        answer_extractor=extract_countdown_answer,
        format_checker=countdown_format_ok,
        convo_prefix=build_llama_convo_prefix(config.model_name, config.prompt_style),
        require_extracted_answer_for_scoring=True,
        logger=logger,
        question_preview_chars=1000,
    )


def main_cli() -> None:
    configure_logging()
    config = chz.entrypoint(InferenceConfig)
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main_cli()
