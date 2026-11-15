"""Mean Teacher (Tarvainen & Valpola 2017) for the shared training loop
(Phase 9, arm21). Default-off: `train_fold` builds one of these only when
`train.ssl` is present in the config; an arm without the block never touches
this module.

The teacher is a deep copy of the student kept in eval() mode (no BatchNorm
anywhere in the SAM 2 + FPN model, so eval() only disables the FPN head's
Dropout2d). Its weights are an exponential moving average of the student's,
taken over the *trainable* parameters only (296K of 74M for LoRA r4): the
frozen base is identical in both copies by construction, so the EMA costs
microseconds and the checkpointed teacher state is ~1 MB.

The teacher is also the model that is evaluated (`eval_model: teacher`, the
convention in the Mean Teacher literature); `student` is the control.
"""
from __future__ import annotations

import copy
import math

import torch

from common.losses.consistency import (confidence_targets, unsup_loss,
                                       unsup_stats)

DEFAULTS = {
    "method": "mean_teacher",
    "ema_alpha": 0.999,
    "alpha_warmup_steps": 0,
    "lambda_u": 1.0,
    "ramp_epochs": 10.0,
    "ramp": "sigmoid",
    "tau_pos": 0.7,
    "tau_neg": 0.1,
    "unsup_terms": [{"name": "bce", "weight": 1.0}, {"name": "dice", "weight": 1.0}],
    "eval_model": "teacher",
    "eval_student_too": False,
    "collapse_max_fg": 0.20,
    "batch_unlabeled": 8,
    "pair_mode": "temporal",
}


def lambda_ramp(t: float, ramp_epochs: float, kind: str = "sigmoid") -> float:
    """Unsupervised weight multiplier at (fractional) epoch `t`.

    sigmoid: exp(-5 (1 - t/T)^2) for t < T, else 1 — the Mean Teacher paper's
    ramp. linear: t/T. T <= 0 means no ramp (always 1).
    """
    if ramp_epochs <= 0:
        return 1.0
    x = min(max(float(t) / float(ramp_epochs), 0.0), 1.0)
    if x >= 1.0:
        return 1.0
    if kind == "sigmoid":
        return float(math.exp(-5.0 * (1.0 - x) ** 2))
    if kind == "linear":
        return float(x)
    raise ValueError(f"unknown ramp {kind!r}")


class MeanTeacher:
    def __init__(self, student: torch.nn.Module, cfg: dict):
        c = {**DEFAULTS, **(cfg or {})}
        if c["method"] != "mean_teacher":
            raise ValueError(f"train.ssl.method must be mean_teacher, got {c['method']!r}")
        if c["eval_model"] not in ("teacher", "student"):
            raise ValueError("train.ssl.eval_model must be teacher|student")
        self.cfg = c
        self.alpha = float(c["ema_alpha"])
        self.alpha_warmup = int(c["alpha_warmup_steps"] or 0)
        self.lambda_u = float(c["lambda_u"])
        self.ramp_epochs = float(c["ramp_epochs"])
        self.ramp = str(c["ramp"])
        self.tau_pos = float(c["tau_pos"])
        self.tau_neg = float(c["tau_neg"])
        self.terms = list(c["unsup_terms"])
        self.eval_model = str(c["eval_model"])
        self.step = 0
        self.names = [n for n, p in student.named_parameters() if p.requires_grad]
        if not self.names:
            raise ValueError("student has no trainable parameters")
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self._t_params = dict(self.teacher.named_parameters())
        self._t_buffers = dict(self.teacher.named_buffers())

    @classmethod
    def from_config(cls, cfg: dict, student) -> "MeanTeacher":
        return cls(student, cfg["train"]["ssl"])

    # ------------------------------------------------------------- weights

    def alpha_at(self, step: int) -> float:
        if self.alpha_warmup > 0:
            return min(self.alpha, (step + 1) / (step + 10))
        return self.alpha

    @torch.no_grad()
    def update(self, student) -> None:
        a = self.alpha_at(self.step)
        s_params = dict(student.named_parameters())
        for n in self.names:
            self._t_params[n].mul_(a).add_(s_params[n].detach(), alpha=1.0 - a)
        s_buf = dict(student.named_buffers())
        for n, b in self._t_buffers.items():
            b.copy_(s_buf[n])
        self.step += 1

    @torch.no_grad()
    def reset_from(self, student) -> None:
        """teacher := student (used when the loop reloads the best state)."""
        self.teacher.load_state_dict(student.state_dict())

    def model_for_eval(self, student):
        return self.teacher if self.eval_model == "teacher" else student

    def train_mode(self) -> None:
        """Called where the loop calls `model.train()`: the teacher stays eval."""
        self.teacher.eval()

    # ---------------------------------------------------------------- loss

    def lambda_at(self, t_epochs: float) -> float:
        return self.lambda_u * lambda_ramp(t_epochs, self.ramp_epochs, self.ramp)

    @torch.no_grad()
    def teacher_probs(self, img_w: torch.Tensor, amp: bool) -> torch.Tensor:
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.teacher(img_w)
        else:
            logits = self.teacher(img_w)
        return torch.sigmoid(logits.float())

    def targets(self, p_t: torch.Tensor, valid: torch.Tensor):
        return confidence_targets(p_t, valid, self.tau_pos, self.tau_neg)

    def unsup_loss(self, logits_u: torch.Tensor, target: torch.Tensor,
                   weight: torch.Tensor):
        return unsup_loss(logits_u, target, weight, self.terms)

    def stats(self, p_t, weight, valid) -> dict:
        return unsup_stats(p_t, weight, valid, self.tau_pos)

    # ----------------------------------------------------------- persistence

    def state_dict(self) -> dict:
        return {"step": self.step, "names": list(self.names),
                "teacher": {n: self._t_params[n].detach().cpu().clone() for n in self.names},
                "buffers": {n: b.detach().cpu().clone() for n, b in self._t_buffers.items()}}

    @torch.no_grad()
    def load_state_dict(self, d: dict) -> None:
        if list(d["names"]) != list(self.names):
            raise ValueError("teacher checkpoint's trainable-parameter set differs "
                             "from the model's (changed peft targets?)")
        for n, v in d["teacher"].items():
            self._t_params[n].copy_(v.to(self._t_params[n].device))
        for n, v in d.get("buffers", {}).items():
            self._t_buffers[n].copy_(v.to(self._t_buffers[n].device))
        self.step = int(d["step"])

    def describe(self) -> dict:
        return {k: v for k, v in self.cfg.items() if k != "strong"} | {
            "n_ema_params": len(self.names), "ema_steps": self.step}
