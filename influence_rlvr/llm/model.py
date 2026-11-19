from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..utils import detect_device
from .config import DEFAULT_MODEL_ID, LoraSpec

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def default_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_base(model_id: str = DEFAULT_MODEL_ID, dtype: torch.dtype | None = None,
              device: torch.device | str | None = None) -> nn.Module:
    from transformers import AutoModelForCausalLM
    dev = detect_device(str(device) if device is not None else None)
    dtype = dtype or default_dtype(dev)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev)
    model.config.use_cache = False
    return model


def add_lora(model: nn.Module, spec: LoraSpec = LoraSpec(), *, gradient_checkpointing: bool = True) -> nn.Module:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=spec.r, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
                     target_modules=list(spec.target_modules), bias=spec.bias, task_type=spec.task_type)
    peft_model = get_peft_model(model, cfg)
    if gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
    return peft_model


def load_adapter(model: nn.Module, checkpoint_dir: str | Path, adapter_name: str = "default") -> nn.Module:
    from peft import load_peft_weights, set_peft_model_state_dict
    weights = load_peft_weights(str(checkpoint_dir), device="cpu")
    set_peft_model_state_dict(model, weights, adapter_name=adapter_name)
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    model.zero_grad(set_to_none=True)
    return model


def build_policy(model_id: str = DEFAULT_MODEL_ID, lora: LoraSpec | None = LoraSpec(), *,
                 device: torch.device | str | None = None, checkpoint_dir: str | Path | None = None,
                 dtype: torch.dtype | None = None, gradient_checkpointing: bool = True,
                 ) -> tuple[nn.Module, PreTrainedTokenizerBase]:
    tok = load_tokenizer(model_id)
    if lora is None:
        model = load_base(model_id, dtype=dtype or torch.float32, device=device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if checkpoint_dir is not None:
            load_weights(model, checkpoint_dir)
        return model, tok
    model = add_lora(load_base(model_id, dtype=dtype, device=device), lora,
                     gradient_checkpointing=gradient_checkpointing)
    if checkpoint_dir is not None:
        load_adapter(model, checkpoint_dir)
    return model, tok


def is_peft(model: nn.Module) -> bool:
    return hasattr(model, "peft_config")


def load_weights(model: nn.Module, checkpoint_dir: str | Path) -> nn.Module:
    from safetensors.torch import load_file
    state = {}
    for f in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        state.update(load_file(str(f), device="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any("lm_head" not in k and "embed" not in k for k in missing):
        raise RuntimeError(f"{checkpoint_dir}: missing {missing[:3]}... unexpected {unexpected[:3]}...")
    model.zero_grad(set_to_none=True)
    return model


def load_checkpoint(model: nn.Module, checkpoint_dir: str | Path) -> nn.Module:
    return load_adapter(model, checkpoint_dir) if is_peft(model) else load_weights(model, checkpoint_dir)


def checkpoint_dirs(run_dir: str | Path, *, require_optimizer: bool = False) -> dict[int, Path]:
    out = {}
    for p in Path(run_dir).glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m is None or not ((p / "adapter_config.json").is_file() or (p / "config.json").is_file()):
            continue
        if require_optimizer and not has_optimizer_state(p):
            continue
        out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def has_optimizer_state(checkpoint_dir: str | Path) -> bool:
    return (Path(checkpoint_dir) / "optimizer.pt").is_file()


def latest_checkpoint(run_dir: str | Path, max_step: int | None = None) -> Path | None:
    dirs = checkpoint_dirs(run_dir)
    steps = [s for s, p in dirs.items()
             if (p / "trainer_state.json").is_file() and s > 0 and (max_step is None or s < max_step)]
    return dirs[max(steps)] if steps else None


def merge_and_save(model_id: str, checkpoint_dir: str | Path, out_dir: str | Path,
                   dtype: torch.dtype = torch.bfloat16) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    merged = PeftModel.from_pretrained(base, str(checkpoint_dir)).merge_and_unload()
    out = Path(out_dir)
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    return out


__all__ = [
    "load_tokenizer", "default_dtype", "load_base", "add_lora", "load_adapter", "build_policy",
    "checkpoint_dirs", "has_optimizer_state", "latest_checkpoint", "merge_and_save",
]
