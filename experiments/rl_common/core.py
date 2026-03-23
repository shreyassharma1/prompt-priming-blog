"""Shared helpers for RL scripts."""

from __future__ import annotations

import logging
from typing import Any

from tinker_cookbook.checkpoint_utils import load_checkpoints_file


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def normalize_stop_reason(stop_reason: Any) -> str | None:
    if stop_reason is None:
        return None
    text = (
        stop_reason.strip()
        if isinstance(stop_reason, str)
        else str(stop_reason).strip()
    )
    return text or None


def inferred_finish_reason_from_parse_success(parse_success: bool) -> str:
    return "stop" if parse_success else "length"


def derive_finish_reason(
    raw_stop_reason: Any, parse_success: bool | None = None
) -> str | None:
    normalized = normalize_stop_reason(raw_stop_reason)
    if normalized is None:
        if parse_success is None:
            return None
        return inferred_finish_reason_from_parse_success(parse_success)

    lowered = normalized.lower()
    if lowered in {"length", "max_tokens", "max_token"}:
        return "length"
    if lowered in {"stop", "eos", "eos_token"}:
        return "stop"
    return normalized


def get_checkpoint_path(
    checkpoint_dir: str,
    step: int | None = None,
    logger: logging.Logger | None = None,
) -> str | None:
    checkpoints = load_checkpoints_file(checkpoint_dir)
    if not checkpoints:
        if logger is not None:
            logger.warning("No checkpoints found in %s", checkpoint_dir)
        return None

    sampler_checkpoints = [
        checkpoint for checkpoint in checkpoints if "sampler_path" in checkpoint
    ]
    if not sampler_checkpoints:
        if logger is not None:
            logger.warning("No sampler checkpoints found")
        return None

    if step is not None:
        for checkpoint in sampler_checkpoints:
            if (
                checkpoint.get("batch") == step
                or checkpoint.get("name") == f"{step:06d}"
            ):
                return checkpoint["sampler_path"]
        if logger is not None:
            logger.warning("No checkpoint found for step %s", step)
        return None

    return sampler_checkpoints[-1]["sampler_path"]
