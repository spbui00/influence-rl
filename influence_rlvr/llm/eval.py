from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch

from .config import EvalSpec
from .generation import hf_generate, tokenize_prompts_batch
from .judge import prefetch_rollout, user_text
from .rewards import MATCH, TEXT_RULES, reward_text

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase


def pass_rate(model: nn.Module, tokenizer: PreTrainedTokenizerBase, targets: Sequence[dict], *, n_samples: int = 16,
              temperature: float = 1.0, top_p: float = 1.0, max_new_tokens: int = 512, seed: int = 12345, batch: int = 8,
              rule: str = MATCH, log: Callable[[str], None] | None = print) -> np.ndarray:
    device = next(model.parameters()).device
    rates = np.zeros(len(targets), dtype=np.float64)
    B = max(1, int(batch))
    model.eval()
    with torch.no_grad():
        for c in range(0, len(targets), B):
            chunk = targets[c:c + B]
            _, ids, mask = tokenize_prompts_batch(tokenizer, [r["prompt"] for r in chunk], device)
            rollout = hf_generate(model, tokenizer, ids, mask, num_samples=n_samples, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_p=top_p, seed=seed + c)
            prefetch_rollout([r["prompt"] for r in chunk], rollout.texts)
            for j, row in enumerate(chunk):
                texts = rollout.texts[j * n_samples:(j + 1) * n_samples]
                r = row.get("reward_rule") if row.get("reward_rule") in TEXT_RULES else rule  # safety rows carry their own rule
                rewards = [reward_text(t, row["gold"], r, user_text(row["prompt"])) for t in texts]
                rates[c + j] = float(np.mean(rewards))
            if log:
                done = min(c + B, len(targets))
                log(f"  [eval] {done}/{len(targets)} targets, running mean {rates[:done].mean():.3f}")
    return rates


def pass_rate_spec(model: nn.Module, tokenizer: PreTrainedTokenizerBase, targets: Sequence[dict],
                   spec: EvalSpec = EvalSpec(), **kw: object) -> np.ndarray:
    return pass_rate(model, tokenizer, targets, n_samples=spec.n_samples, temperature=spec.temperature,
                     top_p=spec.top_p, max_new_tokens=spec.max_new_tokens, seed=spec.seed, batch=spec.batch, **kw)


def write_target_eval(path: str | Path, horizons: dict[int, np.ndarray], extra: dict | None = None) -> dict:
    last = max(horizons)
    payload = {
        "horizons": {str(k): {"mean_reward": float(np.mean(v)), "per_target": [round(float(x), 6) for x in v]}
                     for k, v in sorted(horizons.items())},
        "mean_reward": float(np.mean(horizons[last])),
        "steps": int(last),
        **(extra or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_target_eval(path: str | Path) -> dict[int, np.ndarray]:
    d = json.loads(Path(path).read_text())
    if "horizons" in d:
        out = {}
        for k, v in d["horizons"].items():
            arr = v["per_target"] if isinstance(v, dict) else v
            out[int(k)] = np.asarray(arr, dtype=np.float64)
        return out
    return {int(d.get("steps", 0)): np.asarray(d["per_target"], dtype=np.float64)}


__all__ = ["pass_rate", "pass_rate_spec", "write_target_eval", "read_target_eval"]
