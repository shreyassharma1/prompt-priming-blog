"""Countdown-specific Reasoning Gym RL helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Sequence

import chz
import tinker
from sympy.parsing.sympy_parser import parse_expr
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.utils import logtree

from experiments.rl_common.core import inferred_finish_reason_from_parse_success


FormatChecker = Callable[[str, dict[str, Any]], bool]
AnswerExtractor = Callable[[str, dict[str, Any]], str | None]

_COUNTDOWN_EXPR_RE = re.compile(r"[0-9+\-*/()\s]{3,}")
def _countdown_expression_is_valid(expr: str) -> bool:
    expr = expr.strip().strip("`").strip()
    if not expr:
        return False
    if "=" in expr:
        return False
    if not re.search(r"\d", expr):
        return False
    if not any(op in expr for op in "+-*/"):
        return False
    try:
        parse_expr(expr)
    except Exception:
        return False
    return True


def _extract_countdown_candidate(source: str, target: int | None) -> str | None:
    if not source:
        return None

    normalized_source = source.replace("`", " ")
    candidates: list[str] = []
    target_str = str(target) if target is not None else None

    for raw_line in normalized_source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        candidates.append(line)
        if "=" in line:
            left, right = line.split("=", 1)
            left = left.strip()
            right = right.strip()
            if target_str and target_str in right and target_str not in left:
                candidates.append(left)
            elif target_str and target_str in left and target_str not in right:
                candidates.append(right)
            else:
                candidates.append(left)
                candidates.append(right)

    candidates.append(normalized_source.strip())

    for candidate in reversed(candidates):
        matches = list(_COUNTDOWN_EXPR_RE.finditer(candidate))
        for match in reversed(matches):
            expression = re.sub(r"\s+", " ", match.group(0)).strip(" .,;:")
            if _countdown_expression_is_valid(expression):
                return expression

    return None


def extract_countdown_answer(text: str, entry: dict[str, Any]) -> str | None:
    target_value = entry.get("metadata", {}).get("target")
    target: int | None = None
    if target_value is not None:
        try:
            target = int(float(target_value))
        except (TypeError, ValueError):
            target = None

    return _extract_countdown_candidate(text, target)


def countdown_format_ok(text: str, entry: dict[str, Any]) -> bool:
    answer = extract_countdown_answer(text, entry)
    return bool(answer and _countdown_expression_is_valid(answer))


@dataclass
class ScoreResult:
    score: float
    correct: bool
    extracted_answer: str | None
    scored_answer: str | None


class ReasoningGymEnv(ProblemEnv):
    """ProblemEnv wrapper that scores responses with Reasoning Gym."""

    def __init__(
        self,
        entry: dict[str, Any],
        dataset: Any,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_checker: FormatChecker | None = None,
        answer_extractor: AnswerExtractor | None = None,
        reward_mode: str = "graded",
        format_coef: float = 0.0,
        require_extracted_answer_for_scoring: bool = False,
    ):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.entry = entry
        self.dataset = dataset
        self.format_checker = format_checker
        self.answer_extractor = answer_extractor
        self.reward_mode = reward_mode
        self.require_extracted_answer_for_scoring = require_extracted_answer_for_scoring

    def get_question(self) -> str:
        return self.entry["question"]

    def _score(self, sample_str: str | None) -> ScoreResult:
        if not sample_str:
            return ScoreResult(score=0.0, correct=False, extracted_answer=None, scored_answer=None)

        extracted_answer: str | None = None
        scored_answer: str = sample_str
        if self.answer_extractor is not None:
            try:
                extracted_answer = self.answer_extractor(sample_str, self.entry)
            except Exception:
                extracted_answer = None
            if extracted_answer:
                scored_answer = extracted_answer
            elif self.require_extracted_answer_for_scoring:
                return ScoreResult(
                    score=0.0,
                    correct=False,
                    extracted_answer=None,
                    scored_answer=None,
                )

        try:
            score = float(self.dataset.score_answer(scored_answer, self.entry))
        except Exception:
            score = 0.0
        correct = bool(score >= 1.0)
        return ScoreResult(score=score, correct=correct, extracted_answer=extracted_answer, scored_answer=scored_answer)

    def check_answer(self, sample_str: str) -> bool:
        return self._score(sample_str).correct

    def check_format(self, sample_str: str) -> bool:
        if self.format_checker is None:
            return True
        return self.format_checker(sample_str, self.entry)

    def get_reference_answer(self) -> str:
        return self.entry.get("answer", "")

    async def step(self, action: list[int]) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        parse_success_bool = bool(parse_success)
        content = renderers.get_text_content(message)
        format_ok = parse_success_bool and self.check_format(content)
        score_result = self._score(content)
        if self.reward_mode == "graded":
            reward = score_result.score
        else:
            reward = 1.0 if score_result.correct else 0.0

        if self.format_coef:
            reward += self.format_coef * (float(format_ok) - 1.0)
        finish_reason = inferred_finish_reason_from_parse_success(parse_success_bool)

        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message.get('content', '')}")
        if score_result.extracted_answer is not None:
            logtree.log_text(f"Extracted Answer: {score_result.extracted_answer}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            "Format OK: %s, Correct: %s, Score: %.3f, Reward: %.3f"
            % ("yes" if format_ok else "no", "yes" if score_result.correct else "no", score_result.score, reward)
        )
        logtree.log_text(
            "finish_reason=%s (inferred_from_parse_success), parse_success=%s"
            % (finish_reason, parse_success_bool)
        )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": float(format_ok),
                "correct": float(score_result.correct),
                "score": score_result.score,
                "parse_success": float(parse_success_bool),
                "used_extracted_answer": float(score_result.extracted_answer is not None),
                "finish_reason_stop": float(finish_reason == "stop"),
                "finish_reason_length": float(finish_reason == "length"),
            },
            logs={
                "finish_reason": finish_reason,
                "finish_reason_source": "parse_success_inference",
            },
        )


class ReasoningGymRLDataset(RLDataset):
    def __init__(
        self,
        dataset: Any,
        dataset_name: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None,
        format_checker: FormatChecker | None,
        answer_extractor: AnswerExtractor | None,
        reward_mode: str,
        format_coef: float,
        require_extracted_answer_for_scoring: bool,
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.format_checker = format_checker
        self.answer_extractor = answer_extractor
        self.reward_mode = reward_mode
        self.format_coef = format_coef
        self.require_extracted_answer_for_scoring = require_extracted_answer_for_scoring
        self.size = len(dataset)
        self.num_batches = int(math.ceil(self.size / batch_size))

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.size)
        assert start < end, "Incorrect batch size"
        builders: list[EnvGroupBuilder] = []
        for idx in range(start, end):
            entry = self.dataset[idx]
            builders.append(
                ProblemGroupBuilder(
                    env_thunk=partial(
                        ReasoningGymEnv,
                        entry,
                        self.dataset,
                        self.renderer,
                        self.convo_prefix,
                        self.format_checker,
                        self.answer_extractor,
                        self.reward_mode,
                        self.format_coef,
                        self.require_extracted_answer_for_scoring,
                    ),
                    num_envs=self.group_size,
                    dataset_name=self.dataset_name,
                )
            )
        return builders

    def __len__(self) -> int:
        return self.num_batches


@chz.chz
class ReasoningGymDatasetBuilder(RLDatasetBuilder):
    dataset_factory: Callable[[int, int], Any]
    dataset_name: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    train_size: int
    test_size: int
    seed: int = 0
    convo_prefix: Optional[list[renderers.Message]] = None
    format_checker: FormatChecker | None = None
    answer_extractor: AnswerExtractor | None = None
    reward_mode: str = "graded"
    format_coef: float = 0.0
    require_extracted_answer_for_scoring: bool = False

    async def __call__(self) -> tuple[ReasoningGymRLDataset, ReasoningGymRLDataset | None]:
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_dataset = self.dataset_factory(self.train_size, self.seed)
        train_rl = ReasoningGymRLDataset(
            dataset=train_dataset,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            format_checker=self.format_checker,
            answer_extractor=self.answer_extractor,
            reward_mode=self.reward_mode,
            format_coef=self.format_coef,
            require_extracted_answer_for_scoring=self.require_extracted_answer_for_scoring,
        )

        if self.test_size <= 0:
            return train_rl, None

        test_dataset = self.dataset_factory(self.test_size, self.seed + 1)
        test_rl = ReasoningGymRLDataset(
            dataset=test_dataset,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            group_size=1,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            format_checker=self.format_checker,
            answer_extractor=self.answer_extractor,
            reward_mode=self.reward_mode,
            format_coef=self.format_coef,
            require_extracted_answer_for_scoring=self.require_extracted_answer_for_scoring,
        )
        return train_rl, test_rl
