# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Staged trial evaluation pipeline with prescreen, full eval, validation KL,
generation health, and thinking leak detection.

All configuration lives under ``config.optimization.*``.
"""

import random
import statistics

import torch
import torch.nn.functional as F
from optuna import Trial, TrialPruned

from ..settings import AbliterixConfig
from ..util import print
from .scorer import _finite_logprobs, _safe_kl_divergence
from .screening import TrialScreener


class StageEvaluator:
    """Wraps trial evaluation with a staged pipeline for efficient Optuna
    optimisation.

    Stages:
      1. Prescreen — refusal count on a small fixed subset (default 30).
         Classifies trial as low / borderline / high.
         High → raise TrialPruned.
      2. Prescreen estimation — if low + estimation enabled, skip full eval
         and use a linearly scaled estimate.
      3. Full evaluation — refusal count on all target_msgs.
      4. Validation KL — KL divergence on a held-out benign subset.
      5. Generation health — ngram repetition, token frequency, etc.
      6. Thinking leak — detect chain-of-thought markers.
    """

    def __init__(self, config: AbliterixConfig, engine, scorer, prescreen_seed: int = 117):
        self.config = config
        self.engine = engine
        self.scorer = scorer
        opt = config.optimization

        # Pre-select prescreen indices from target_msgs (random, seeded, same for all trials).
        n_target = len(scorer.target_msgs)
        if n_target > 0:
            rng = random.Random(opt.refusal_prescreen_seed)
            prescreen_n = min(opt.refusal_prescreen_size, n_target)
            self._prescreen_indices = rng.sample(range(n_target), prescreen_n)
            self._prescreen_msgs = [scorer.target_msgs[i] for i in self._prescreen_indices]
        else:
            self._prescreen_indices = []
            self._prescreen_msgs = []

        # Pre-select validation KL prompts from benign eval msgs.
        if opt.validation_kl_enabled:
            n_benign = len(scorer.benign_msgs)
            if n_benign > 0:
                rng2 = random.Random(opt.refusal_prescreen_seed + 1)
                val_n = min(opt.validation_kl_size, n_benign)
                self._validation_indices = rng2.sample(range(n_benign), val_n)
                self._validation_msgs = [scorer.benign_msgs[i] for i in self._validation_indices]
            else:
                self._validation_indices = []
                self._validation_msgs = []
        else:
            self._validation_indices = []
            self._validation_msgs = []

    def evaluate(self, trial: Trial, kl: float, benign_responses: list[str],
                 skip_prescreen: bool = False, prescreen_result: tuple | None = None) -> tuple:
        """Run the full staged evaluation pipeline.

        Parameters
        ----------
        trial : Trial
            The Optuna trial being evaluated.
        kl : float
            KL divergence already measured by ``scorer.measure_kl_and_coherence``.
        benign_responses : list[str]
            Generated benign responses (from the KL/coherence pass).
        skip_prescreen : bool
            If True, skip the prescreen stage entirely. Use this when the
            prescreen has already been run separately (e.g., before KL
            measurement to avoid wasting compute on a doomed trial).
        prescreen_result : tuple | None
            Cached ``(detected_full, skip_full_eval)`` from a prior prescreen.
            Required when ``skip_prescreen=True``.

        Returns
        -------
        (detected_full, screening_report)
            ``detected_full`` is the refusal count (estimated or full).
            ``screening_report`` is a dict with gen health and thinking leak results.

        Raises
        ------
        TrialPruned
            If the prescreen classifies the trial as high-refusal.
        """
        opt = self.config.optimization
        compliance_objective: float | None = None

        # ----------------------------------------------------------------
        # Stage 1 & 2: Prescreen + estimation
        # ----------------------------------------------------------------
        if not opt.refusal_prescreen_enabled:
            detected_full, skip_full_eval = 0, False
        elif skip_prescreen:
            if prescreen_result is None:
                raise ValueError(
                    "prescreen_result must be provided when skip_prescreen=True"
                )
            detected_full, skip_full_eval = prescreen_result
        else:
            detected_full, skip_full_eval = self._run_prescreen(trial)

        # ----------------------------------------------------------------
        # Stage 3: Full evaluation (if not skipped by prescreen estimation)
        # ----------------------------------------------------------------
        if not skip_full_eval:
            measure_compliance = getattr(
                self.scorer, "measure_compliance_objective", None
            )
            if callable(measure_compliance):
                detected_full, compliance_objective = measure_compliance(self.engine)
            else:
                print("  * Counting model refusals...")
                detected_full = self.scorer.detector.evaluate_compliance(
                    self.engine,
                    self.scorer.target_msgs,
                )
            trial.set_user_attr("refusals_score_source", "full")

        # Color-coded refusal output
        n_target = len(self.scorer.target_msgs)
        if float(detected_full) / max(n_target, 1) <= 0.08:
            ref_color = "green"
        elif float(detected_full) / max(n_target, 1) <= 0.25:
            ref_color = "yellow"
        else:
            ref_color = "red"
        label = "Refusals (est)" if skip_full_eval else "Refusals"
        print(f"  * {label}: [{ref_color}]{detected_full}/{n_target}[/]")

        # ----------------------------------------------------------------
        # Stage 4: Validation KL
        # ----------------------------------------------------------------
        if not skip_full_eval and opt.validation_kl_enabled and self._validation_msgs:
            self._run_validation_kl(trial)

        # ----------------------------------------------------------------
        # Stage 5 & 6: Generation health + Thinking leak
        # ----------------------------------------------------------------
        screening_report: dict = {}

        if not skip_full_eval:
            screener = TrialScreener(self.config, self.engine)

            if opt.generation_health_enabled and benign_responses:
                health = screener.check_generation_health(benign_responses)
                trial.set_user_attr("generation_health", health)
                screener.print_screening_report(health)
                screening_report["generation_health"] = health

            if opt.thinking_leak_detection_enabled and benign_responses:
                leaked = screener.check_thinking_leak(benign_responses)
                trial.set_user_attr("thinking_leak_detected", leaked)
                if leaked:
                    print("  * Thinking leak: [yellow]detected[/]")
                    if "generation_health" in screening_report:
                        screening_report["generation_health"]["thinking_leak_detected"] = True
                else:
                    print("  * Thinking leak: [green]none[/]")
                screening_report["thinking_leak_detected"] = leaked

        # Expose compliance override for upstream _compute_objectives when available.
        screening_report["compliance_objective"] = compliance_objective
        return detected_full, screening_report

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _run_prescreen(self, trial: Trial) -> tuple:
        """Run the refusal prescreen on a fixed prompt subset.

        Returns ``(detected_full, skip_full_eval)``.
        May raise ``TrialPruned`` for high-refusal trials.
        """
        opt = self.config.optimization

        if not self._prescreen_msgs:
            # No prescreen prompts available — fall through to full eval.
            return 0, False

        detected_30 = self.scorer.detector.evaluate_compliance(
            self.engine, self._prescreen_msgs,
        )
        trial.set_user_attr("prescreen_refusals", detected_30)
        trial.set_user_attr("prescreen_indices", self._prescreen_indices)

        if detected_30 >= opt.refusal_prescreen_prune_min:
            trial.set_user_attr("prescreen_class", "high")
            print(
                f"  * Refusal prescreen: {detected_30}/{len(self._prescreen_msgs)} "
                "[red]high[/] (prune)"
            )
            raise TrialPruned()

        elif detected_30 <= opt.refusal_prescreen_pass_max:
            trial.set_user_attr("prescreen_class", "low")
            print(
                f"  * Refusal prescreen: {detected_30}/{len(self._prescreen_msgs)} "
                "[green]low[/]"
            )
        else:
            trial.set_user_attr("prescreen_class", "borderline")
            ratio = len(self.scorer.target_msgs) / len(self._prescreen_msgs)
            estimated = round(detected_30 * ratio)
            trial.set_user_attr("estimated_full_refusals", estimated)
            trial.set_user_attr("refusals_score_source", "prescreen_estimate")
            print(
                f"  * Refusal prescreen: {detected_30}/{len(self._prescreen_msgs)} "
                "[yellow]borderline[/] (estimate ~{}/{})".format(
                    estimated, len(self.scorer.target_msgs)
                )
            )
            return estimated, True
        # Low → proceed to full evaluation.
        return 0, False

    def _run_validation_kl(self, trial: Trial) -> None:
        """Measure KL divergence on a held-out validation set of benign prompts.

        Computes per-prompt KL values and stores mean/p95/max + top-1 disagreement.
        """
        opt = self.config.optimization
        if not self._validation_msgs:
            return

        vllm_gen = getattr(self.engine, "_vllm_gen", None)
        adapter_path = getattr(self.engine, "_current_adapter_path", None)

        # Fixed baseline continuations for the validation subset.  Both the
        # baseline and the steered model must be scored on the SAME
        # teacher-forced prefixes, otherwise the KL explodes once the
        # steered model's free-run trajectory diverges from baseline (which
        # is exactly what a successful trial does).  Using the baseline
        # continuations for both sides keeps every KL step prefix-identical.
        baseline_cont = getattr(self.scorer, "baseline_continuations", None)
        if baseline_cont is None or self.scorer.baseline_logprobs is None:
            print(
                "  [yellow]validation KL skipped: baseline continuations "
                "unavailable[/]"
            )
            return
        baseline_cont = baseline_cont[self._validation_indices]

        if vllm_gen is not None:
            v_logprobs = vllm_gen.score_continuation_logprobs_batched(
                self._validation_msgs,
                baseline_cont,
                self.config.kl.token_count,
                adapter_path=adapter_path,
            )
        else:
            v_logprobs = self.engine.score_continuation_logprobs_batched(
                self._validation_msgs,
                baseline_cont,
                self.config.kl.token_count,
            )

        # Baseline logprobs for just the validation subset.
        baseline_v = self.scorer.baseline_logprobs[self._validation_indices]

        cur_lp = _finite_logprobs(v_logprobs)
        base_lp = _finite_logprobs(baseline_v)
        if base_lp.device != cur_lp.device:
            base_lp = base_lp.to(cur_lp.device)

        # Multi-token capture returns (batch, step, vocab); single-token keeps
        # (batch, vocab).  Sum KL over vocab, then average over the step axis
        # (same per-token averaging as _safe_kl_divergence) → per-prompt KL.
        kl_none = F.kl_div(cur_lp, base_lp, reduction="none", log_target=True)
        kl_per_tensor = kl_none.sum(dim=-1)
        if kl_per_tensor.dim() > 1:
            kl_per_tensor = kl_per_tensor.mean(dim=-1)
        kl_per_prompt = kl_per_tensor.cpu().tolist()
        if isinstance(kl_per_prompt, float):
            kl_per_prompt = [kl_per_prompt]

        # Top-1 disagreement: use single-token logprobs (Ornith method).
        # Averaged logprobs flatten the distribution → argmax flips too easily.
        # Distinguish low-margin flips (near-tied top-2) from high-margin flips
        # (confident top-1 replaced) to avoid noise from argmax discreteness.
        cur_single = self.engine._logprobs_forward_pass(self._validation_msgs)
        base_single = self.scorer.baseline_single_token_lp[self._validation_indices]
        if base_single.device != cur_single.device:
            base_single = base_single.to(cur_single.device)

        base_top1 = base_single.argmax(dim=-1)
        cur_top1 = cur_single.argmax(dim=-1)
        disagree_mask = cur_top1 != base_top1

        # Margin: difference between top-1 and top-2 logprob in base model.
        # Low margin (< 0.5 nats) means top-2 were near-tied, so a flip is
        # statistically meaningless even if argmax changed.
        base_top2 = base_single.topk(2, dim=-1).values
        base_margin = (base_top2[:, 0] - base_top2[:, 1]).cpu()

        disagree_flags = disagree_mask.float().cpu()
        low_margin_thresh = 1.0
        low_margin_flips = int(((disagree_flags == 1) & (base_margin < low_margin_thresh)).sum())
        high_margin_flips = int(((disagree_flags == 1) & (base_margin >= low_margin_thresh)).sum())
        total_disagree = low_margin_flips + high_margin_flips
        n_val = int(disagree_flags.numel())
        mean_disagree = total_disagree / n_val if n_val else 0.0
        high_margin_rate = high_margin_flips / n_val if n_val else 0.0

        mean_kl = statistics.mean(kl_per_prompt) if kl_per_prompt else 0.0
        sorted_kl = sorted(kl_per_prompt)
        p95_idx = int(0.95 * len(sorted_kl))
        p95_kl = sorted_kl[min(p95_idx, len(sorted_kl) - 1)] if sorted_kl else 0.0
        max_kl = sorted_kl[-1] if sorted_kl else 0.0

        trial.set_user_attr("validation_kl_mean", mean_kl)
        trial.set_user_attr("validation_kl_p95", p95_kl)
        trial.set_user_attr("validation_kl_max", max_kl)
        trial.set_user_attr("validation_top1_disagreement_rate", high_margin_rate)
        trial.set_user_attr("validation_top1_disagreement_count", total_disagree)
        trial.set_user_attr("validation_top1_low_margin_flips", low_margin_flips)
        trial.set_user_attr("validation_top1_high_margin_flips", high_margin_flips)
        trial.set_user_attr("validation_top1_total_rate", mean_disagree)
        trial.set_user_attr("validation_sample_count", n_val)

        vq = "green" if mean_kl < 0.1 else ("yellow" if mean_kl < 0.5 else "red")
        dq = "green" if high_margin_rate < 0.05 else ("yellow" if high_margin_rate < 0.15 else "red")
        print(
            f"  * Validation KL: [{vq}]{mean_kl:.4f}[/]"
            f"  p95={p95_kl:.4f}  max={max_kl:.4f}"
            f"  top1_high_margin=[{dq}]{high_margin_flips}/{n_val} ({high_margin_rate * 100:.1f}%)[/]"
            f"  total={total_disagree} low_margin={low_margin_flips}"
        )
