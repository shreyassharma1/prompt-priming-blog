"""Shared train/inference runners for RL task."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.train import Config as TrainCoreConfig
from tinker_cookbook.rl.train import main as train_main
from tinker_cookbook.tokenizer_utils import get_tokenizer

from experiments.reasoning_gym.rl_env import ReasoningGymDatasetBuilder
from experiments.rl_common.core import (
    derive_finish_reason,
    get_checkpoint_path,
    normalize_stop_reason,
)


@dataclass
class InferenceResult:
    entry_idx: int
    entry: dict[str, Any]
    prompt_messages: list[renderers.Message]
    prompt_text: str
    response: str
    response_text: str
    extracted_answer: str | None
    scored_answer: str
    score: float
    is_correct: bool
    format_ok: bool
    model_type: str
    raw_stop_reason: str | None
    finish_reason: str | None


def _default_run_name(run_name: str | None) -> str:
    if run_name is not None:
        return run_name
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def build_prompt_messages(
    entry: dict[str, Any], convo_prefix: list[renderers.Message] | None
) -> list[renderers.Message]:
    messages: list[renderers.Message] = []
    if convo_prefix:
        messages.extend(convo_prefix)
    messages.append({"role": "user", "content": entry["question"]})
    return messages


def format_prompt_messages(messages: list[renderers.Message]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content", "")
        rendered = renderers.format_content_as_string(content)
        lines.append(f"[{role}]")
        lines.append(rendered)
    return "\n\n".join(lines)


def _serialize_config(config: Any) -> dict[str, Any]:
    if not hasattr(config, "__chz_fields__"):
        raise TypeError(f"Config must be a chz config instance; got {type(config)!r}")

    field_names = list(config.__chz_fields__.keys())
    result: dict[str, Any] = {}

    for field_name in field_names:
        value = getattr(config, field_name)
        if isinstance(value, tuple):
            value = list(value)
        result[field_name] = value

    return result


async def run_reasoning_gym_training(
    config: Any,
    *,
    runs_dir: Path,
    dataset_name: str,
    dataset_factory: Callable[[Any, int, int], Any],
    answer_extractor: Callable[[str, dict[str, Any]], str | None] | None,
    format_checker: Callable[[str, dict[str, Any]], bool] | None,
    convo_prefix: list[renderers.Message] | None,
    logger,
    require_extracted_answer_for_scoring: bool = False,
) -> str:
    run_name = _default_run_name(getattr(config, "run_name", None))
    output_dir = str(runs_dir / run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    task_config_path = os.path.join(output_dir, "task_config.json")
    task_config_data = _serialize_config(config)
    task_config_data["dataset_name"] = dataset_name
    with open(task_config_path, "w") as f:
        json.dump(task_config_data, f, indent=2)
    logger.info("Task config saved to %s", task_config_path)

    renderer_name = getattr(config, "renderer_name", None)
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    logger.info("Using renderer: %s", renderer_name)

    dataset_builder = ReasoningGymDatasetBuilder(
        dataset_factory=lambda size, seed: dataset_factory(config, size, seed),
        dataset_name=dataset_name,
        batch_size=config.groups_per_batch,
        group_size=config.group_size,
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        train_size=config.train_size,
        test_size=config.test_size,
        seed=config.seed,
        convo_prefix=convo_prefix,
        format_checker=format_checker,
        answer_extractor=answer_extractor,
        reward_mode=config.reward_mode,
        format_coef=config.format_coef,
        require_extracted_answer_for_scoring=require_extracted_answer_for_scoring,
    )

    train_config = TrainCoreConfig(
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        loss_fn=getattr(config, "loss_fn", "importance_sampling"),
        loss_fn_config=getattr(config, "loss_fn_config", None),
        num_substeps=getattr(config, "num_substeps", 1),
        temperature=getattr(config, "temperature", 1.0),
        compute_post_kl=getattr(config, "compute_post_kl", False),
        remove_constant_reward_groups=getattr(
            config, "remove_constant_reward_groups", False
        ),
        log_path=output_dir,
        save_every=config.save_every,
        eval_every=config.eval_every,
        wandb_project=None,
        wandb_name=None,
    )

    await train_main(train_config)

    logger.info("Training complete! Checkpoints saved to: %s", output_dir)
    return output_dir


async def run_inference_on_entry(
    prompt_messages: list[renderers.Message],
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    *,
    max_tokens: int | None,
    temperature: float,
    n_samples: int,
) -> list[tuple[str, str, str | None, str | None]]:
    model_input = renderer.build_generation_prompt(prompt_messages)
    stop_sequences = renderer.get_stop_sequences()

    result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=n_samples,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
        ),
    )

    responses: list[tuple[str, str, str | None, str | None]] = []
    for seq in result.sequences:
        raw_stop_reason = normalize_stop_reason(getattr(seq, "stop_reason", None))
        parsed_message, parse_success = renderer.parse_response(seq.tokens)
        finish_reason = derive_finish_reason(
            raw_stop_reason, parse_success=bool(parse_success)
        )
        content = parsed_message.get("content", "")
        response_full = renderers.format_content_as_string(content)
        response_text = renderers.get_text_content(parsed_message)
        responses.append((response_full, response_text, raw_stop_reason, finish_reason))

    return responses


async def run_reasoning_gym_inference(
    entries: list[dict[str, Any]],
    dataset: Any,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    *,
    model_type: str,
    answer_extractor: Callable[[str, dict[str, Any]], str | None],
    format_checker: Callable[[str, dict[str, Any]], bool],
    max_tokens: int | None,
    temperature: float,
    n_samples: int,
    convo_prefix: list[renderers.Message] | None,
    logger,
    require_extracted_answer_for_scoring: bool = False,
    progress_every: int = 10,
    max_concurrency: int = 1,
) -> list[InferenceResult]:
    results: list[InferenceResult] = []

    async def process_entry(
        entry_idx: int, entry: dict[str, Any]
    ) -> tuple[int, list[InferenceResult]]:
        prompt_messages = build_prompt_messages(entry, convo_prefix)
        prompt_text = format_prompt_messages(prompt_messages)
        responses = await run_inference_on_entry(
            prompt_messages,
            sampling_client,
            renderer,
            max_tokens=max_tokens,
            temperature=temperature,
            n_samples=n_samples,
        )

        entry_results: list[InferenceResult] = []
        for response_full, response_text, raw_stop_reason, finish_reason in responses:
            extracted_answer = answer_extractor(response_text, entry)
            if extracted_answer is not None:
                scored_answer = extracted_answer
                score = float(dataset.score_answer(scored_answer, entry))
            elif require_extracted_answer_for_scoring:
                scored_answer = ""
                score = 0.0
            else:
                scored_answer = response_text
                score = float(dataset.score_answer(scored_answer, entry))
            is_correct = score >= 1.0
            format_ok = format_checker(response_text, entry)
            entry_results.append(
                InferenceResult(
                    entry_idx=entry_idx,
                    entry=entry,
                    prompt_messages=prompt_messages,
                    prompt_text=prompt_text,
                    response=response_full,
                    response_text=response_text,
                    extracted_answer=extracted_answer,
                    scored_answer=scored_answer,
                    score=score,
                    is_correct=is_correct,
                    format_ok=format_ok,
                    model_type=model_type,
                    raw_stop_reason=raw_stop_reason,
                    finish_reason=finish_reason,
                )
            )
        return entry_idx, entry_results

    if max_concurrency <= 1:
        for i, entry in enumerate(entries):
            _, entry_results = await process_entry(i, entry)
            results.extend(entry_results)
            if (i + 1) % progress_every == 0 or i + 1 == len(entries):
                mean_score = (
                    sum(r.score for r in results) / len(results) if results else 0.0
                )
                acc = (
                    sum(1 for r in results if r.is_correct) / len(results)
                    if results
                    else 0.0
                )
                logger.info(
                    "[%s] Processed %d/%d (%d samples), score=%.3f acc=%.2f%%",
                    model_type,
                    i + 1,
                    len(entries),
                    len(results),
                    mean_score,
                    acc * 100,
                )
    else:
        processed = 0
        for batch_start in range(0, len(entries), max_concurrency):
            batch = list(
                enumerate(
                    entries[batch_start : batch_start + max_concurrency],
                    start=batch_start,
                )
            )
            tasks = [
                asyncio.create_task(process_entry(entry_idx, entry))
                for entry_idx, entry in batch
            ]
            for task in asyncio.as_completed(tasks):
                _, entry_results = await task
                results.extend(entry_results)
                processed += 1
                if processed % progress_every == 0 or processed == len(entries):
                    mean_score = sum(r.score for r in results) / len(results)
                    acc = sum(1 for r in results if r.is_correct) / len(results)
                    logger.info(
                        "[%s] Processed %d/%d (%d samples), score=%.3f acc=%.2f%%",
                        model_type,
                        processed,
                        len(entries),
                        len(results),
                        mean_score,
                        acc * 100,
                    )

    return results


def print_reasoning_gym_results(
    results: list[InferenceResult], title: str, *, question_preview_chars: int = 1200
) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    mean_score = sum(r.score for r in results) / len(results)
    acc = sum(1 for r in results if r.is_correct) / len(results)
    fmt_ok = sum(1 for r in results if r.format_ok) / len(results)
    raw_stop_reason_length = sum(1 for r in results if r.raw_stop_reason == "length")
    finish_reason_length = sum(1 for r in results if r.finish_reason == "length")
    print(f"Mean score: {mean_score:.3f}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Format OK: {fmt_ok:.2%}")
    print(
        "Raw stop reason = length: "
        f"{raw_stop_reason_length}/{len(results)} = {raw_stop_reason_length / len(results):.2%}"
    )
    print(
        "Finish reason = length: "
        f"{finish_reason_length}/{len(results)} = {finish_reason_length / len(results):.2%}\n"
    )

    print("Sample outputs:")
    print("-" * 80)
    for result in results[:5]:
        status = "OK" if result.is_correct else "X"
        print(
            f"{status} Score={result.score:.3f} Format={'ok' if result.format_ok else 'bad'}"
        )
        print(f"Raw stop/Finish: {result.raw_stop_reason!r} / {result.finish_reason!r}")
        print(f"Extracted: {result.extracted_answer!r}")
        print("Prompt:")
        print(result.prompt_text[:question_preview_chars])
        print("Q:")
        print(result.entry["question"][:question_preview_chars])
        print("A:")
        print(result.response[:600])
        print()


async def run_reasoning_gym_inference_cli(
    config: Any,
    *,
    dataset_name: str,
    dataset_factory: Callable[[Any, int, int], Any],
    answer_extractor: Callable[[str, dict[str, Any]], str | None],
    format_checker: Callable[[str, dict[str, Any]], bool],
    convo_prefix: list[renderers.Message] | None,
    logger,
    require_extracted_answer_for_scoring: bool = False,
    question_preview_chars: int = 1200,
) -> list[InferenceResult]:
    dataset = dataset_factory(config, size=config.n_problems, seed=config.seed)
    entries = [dataset[i] for i in range(config.n_problems)]
    logger.info("Loaded %d %s problems", len(entries), dataset_name)
    max_concurrency = max(1, int(getattr(config, "max_concurrency", 1)))

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = getattr(config, "renderer_name", None)
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()

    logger.info(
        "Sampling with temperature=%s, n_samples=%s, max_concurrency=%s",
        config.temperature,
        config.n_samples,
        max_concurrency,
    )
    logger.info("Using renderer: %s", renderer_name)

    checkpoint_path = get_checkpoint_path(
        config.checkpoint_dir, config.step, logger=logger
    )
    if checkpoint_path is None:
        raise ValueError(f"Could not find checkpoint in {config.checkpoint_dir}")

    logger.info("Running inference on trained model from: %s", checkpoint_path)
    sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
    results = await run_reasoning_gym_inference(
        entries,
        dataset,
        sampling_client,
        renderer,
        model_type="trained",
        answer_extractor=answer_extractor,
        format_checker=format_checker,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        n_samples=config.n_samples,
        convo_prefix=convo_prefix,
        require_extracted_answer_for_scoring=require_extracted_answer_for_scoring,
        logger=logger,
        max_concurrency=max_concurrency,
    )
    print_reasoning_gym_results(
        results,
        f"TRAINED MODEL (step {config.step or 'final'}, prompt={config.prompt_style}) [temp={config.temperature}]",
        question_preview_chars=question_preview_chars,
    )

    if config.output_file:
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = [
            {
                "question": r.entry["question"],
                "prompt_messages": r.prompt_messages,
                "prompt_text": r.prompt_text,
                "expected_answer": r.entry.get("answer", ""),
                "response": r.response,
                "response_text": r.response_text,
                "extracted_answer": r.extracted_answer,
                "scored_answer": r.scored_answer,
                "score": r.score,
                "is_correct": r.is_correct,
                "format_ok": r.format_ok,
                "model_type": r.model_type,
                "raw_stop_reason": r.raw_stop_reason,
                "finish_reason": r.finish_reason,
                "metadata": r.entry.get("metadata", {}),
            }
            for r in results
        ]
        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return results
