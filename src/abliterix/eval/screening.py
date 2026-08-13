# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-trial generation health screening: ngram repetition, token frequency,
consecutive repeats, and thinking leak detection.

The :class:`TrialScreener` runs diagnostic checks on already-generated responses
and does not require any model inference.
"""

import collections
import re
from typing import Any

from ..util import print


def _ngram_iter(tokens: list[int], n: int):
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


class TrialScreener:
    """Post-trial diagnostic checks on generated responses.

    All methods operate on already-generated text and return diagnostic dicts
    that are stored as Optuna trial user attributes.
    """

    def __init__(self, config, engine):
        self.config = config
        self.tokenizer = engine.tokenizer
        self._thinking_skip_patterns = [
            "<think>",
            "</think>",
        ]
        # Plain words such as "thinking" and "thought" occur routinely in
        # benign prose (critical-thinking lessons, story brainstorming, etc.).
        # Treat them as a leak only when rendered as a standalone reasoning
        # section header, while explicit model control tags remain substring
        # matches above.
        self._thinking_header_re = re.compile(
            r"(?im)^\s*(?:thinking|thought|思考)\s*[:：]\s*"
        )

    def check_generation_health(self, responses: list[str]) -> dict[str, Any]:
        """Analyse ngram repetition, token frequency, and consecutive repeats.

        Returns a dict with per-response aggregates and a ``passed`` flag.
        """
        health = {
            "repetition_ratio": 0.0,
            "distinct_2": 0,
            "distinct_4": 0,
            "max_ngram_repeat": 0,
            "token_max_freq_ratio": 0.0,
            "consecutive_max": 0,
            "digit_ratio": 0.0,
            "passed": True,
            "reasons": [],
        }

        if not responses or not any(r.strip() for r in responses):
            health["passed"] = False
            health["reasons"].append("empty responses")
            return health

        repetition_ratios: list[float] = []
        distinct_n: dict[int, list[int]] = {2: [], 4: []}
        max_ngram_repeats: list[int] = []
        token_freq_ratios: list[float] = []
        consecutive_maxs: list[int] = []
        digit_ratios: list[float] = []

        for resp in responses:
            ids = self.tokenizer.encode(resp)
            if not ids:
                repetition_ratios.append(0.0)
                distinct_n[2].append(0)
                distinct_n[4].append(0)
                max_ngram_repeats.append(0)
                token_freq_ratios.append(0.0)
                consecutive_maxs.append(0)
                digit_ratios.append(0.0)
                continue

            n_total = len(ids)
            n_unique = len(set(ids))
            repetition_ratios.append(
                1.0 - n_unique / n_total if n_total > 0 else 0.0
            )

            for n in (2, 4):
                ngrams = set(_ngram_iter(ids, n))
                distinct_n[n].append(len(ngrams))

            ngram_counts: dict[tuple[int, ...], int] = collections.Counter()
            for n in range(1, 5):
                for ng in _ngram_iter(ids, n):
                    ngram_counts[ng] += 1
            max_ngram_repeats.append(
                max(ngram_counts.values()) if ngram_counts else 0
            )

            token_counts = collections.Counter(ids)
            max_freq = max(token_counts.values()) if token_counts else 0
            token_freq_ratios.append(
                max_freq / n_total if n_total > 0 else 0.0
            )

            cons = 1
            best = 1
            for j in range(1, len(ids)):
                if ids[j] == ids[j - 1]:
                    cons += 1
                    best = max(best, cons)
                else:
                    cons = 1
            consecutive_maxs.append(best)

            n_digits = sum(1 for ch in resp if ch.isdigit())
            digit_ratios.append(n_digits / len(resp) if resp else 0.0)

        health["repetition_ratio"] = (
            sum(repetition_ratios) / len(repetition_ratios)
            if repetition_ratios
            else 0.0
        )
        health["distinct_2"] = sum(distinct_n[2])
        health["distinct_4"] = sum(distinct_n[4])
        health["max_ngram_repeat"] = max(max_ngram_repeats) if max_ngram_repeats else 0
        health["token_max_freq_ratio"] = (
            max(token_freq_ratios) if token_freq_ratios else 0.0
        )
        health["consecutive_max"] = max(consecutive_maxs) if consecutive_maxs else 0
        health["digit_ratio"] = (
            max(digit_ratios) if digit_ratios else 0.0
        )

        reasons: list[str] = []
        if health["repetition_ratio"] > 0.6:
            reasons.append(
                f"repetition_ratio={health['repetition_ratio']:.3f} > 0.6"
            )
        if health["max_ngram_repeat"] > 15:
            reasons.append(
                f"max_ngram_repeat={health['max_ngram_repeat']} > 15"
            )
        if health["consecutive_max"] > 30:
            reasons.append(
                f"consecutive_max={health['consecutive_max']} > 30"
            )
        if health["token_max_freq_ratio"] > 0.85:
            reasons.append(
                f"token_max_freq_ratio={health['token_max_freq_ratio']:.3f} > 0.85"
            )
        if health["digit_ratio"] > 0.15:
            reasons.append(
                f"digit_ratio={health['digit_ratio']:.3f} > 0.15"
            )
        health["reasons"] = reasons
        health["passed"] = len(reasons) == 0

        return health

    def check_thinking_leak(
        self,
        responses: list[str],
        extra_patterns: list[str] | None = None,
    ) -> bool:
        """Return True if any response contains chain-of-thought markers."""
        patterns = list(self._thinking_skip_patterns)
        if extra_patterns:
            patterns.extend(extra_patterns)
        for resp in responses:
            lower = resp.lower()
            for pattern in patterns:
                if pattern.lower() in lower:
                    return True
            if self._thinking_header_re.search(resp):
                return True
        return False

    def print_screening_report(self, report: dict[str, Any]) -> None:
        """Print a colour-coded Rich summary of a generation-health report."""
        passed = report.get("passed", True)
        badge = "[green]PASSED[/]" if passed else "[red]FAILED[/]"
        print(f"  * Generation health: {badge}")

        def _color(val, lo, hi):
            if val <= lo:
                return "green"
            if val <= hi:
                return "yellow"
            return "red"

        fields = [
            ("Repetition ratio", "repetition_ratio", 0.4, 0.6, "{:.3f}"),
            ("Distinct 2-grams", "distinct_2", -1, -1, "{}"),
            ("Distinct 4-grams", "distinct_4", -1, -1, "{}"),
            ("Max ngram repeat", "max_ngram_repeat", 8, 15, "{}"),
            ("Token freq ratio", "token_max_freq_ratio", 0.6, 0.85, "{:.3f}"),
            ("Consecutive max", "consecutive_max", 15, 30, "{}"),
        ]

        for label, key, lo, hi, fmt in fields:
            val = report.get(key, 0)
            if isinstance(val, float) and lo >= 0:
                col = _color(val, lo, hi)
                print(
                    f"    [{col}]• {label}: {fmt.format(val)}[/]"
                )
            else:
                print(f"    • {label}: {fmt.format(val)}")

        reasons = report.get("reasons", [])
        if reasons:
            for r in reasons:
                print(f"    [red]  reason: {r}[/]")

        if report.get("thinking_leak_detected", False):
            print("    [yellow]  thinking leak detected[/]")
