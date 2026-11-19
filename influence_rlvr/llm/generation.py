"""HF rollouts keep the EOS token inside the response (its log-prob is part of every gradient) and
cut at the first pad only when pad != eos; the vLLM-server path takes the engine's completion ids
as-is. torch.manual_seed(seed) is set globally right before generate (legacy determinism)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

Tensor = torch.Tensor
Prompt = str | Sequence[dict]


@dataclass
class RolloutBatch:
    texts: list[str]
    token_ids: Tensor
    response_mask: Tensor
    num_prompts: int = 1
    num_samples: int = 1

    def prompt_slice(self, i: int) -> "RolloutBatch":
        s = slice(i * self.num_samples, (i + 1) * self.num_samples)
        return RolloutBatch(self.texts[s], self.token_ids[s], self.response_mask[s], 1, self.num_samples)


def rollout_to_completions(rollout: RolloutBatch) -> list[list[dict]]:
    return [[{"role": "assistant", "content": t}] for t in rollout.texts]


def render_prompt(tokenizer: PreTrainedTokenizerBase, prompt: Prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(list(prompt), tokenize=False, add_generation_prompt=True)


def tokenize_prompt(tokenizer: PreTrainedTokenizerBase, prompt: Prompt, device: torch.device | str) -> tuple[str, Tensor, Tensor]:
    text = render_prompt(tokenizer, prompt)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device) if "attention_mask" in enc else torch.ones_like(ids)
    return text, ids, mask


def tokenize_prompts_batch(tokenizer: PreTrainedTokenizerBase, prompts: Sequence[Prompt], device: torch.device | str,
                           padding_side: str = "left") -> tuple[list[str], Tensor, Tensor]:
    texts = [render_prompt(tokenizer, p) for p in prompts]
    prev = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tokenizer.padding_side = prev
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device) if "attention_mask" in enc else torch.ones_like(ids)
    return texts, ids, mask


def pad_id(tokenizer: PreTrainedTokenizerBase) -> int:
    for v in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0):
        if v is not None:
            return int(v)
    return 0


def stack_sequences(seqs: Sequence[Tensor], pad: int, device: torch.device | str) -> tuple[Tensor, Tensor]:
    if not seqs:
        raise RuntimeError("no response tokens were generated")
    max_len = max(int(s.numel()) for s in seqs)
    if max_len <= 0:
        raise RuntimeError("no response tokens were generated")
    ids = torch.full((len(seqs), max_len), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        n = int(s.numel())
        ids[i, :n] = s.reshape(-1).to(device=device, dtype=torch.long)
        mask[i, :n] = 1
    return ids, mask


def rollout_from_sequences(tokenizer: PreTrainedTokenizerBase, seqs: Sequence[Tensor], *, device: torch.device | str,
                           num_prompts: int, num_samples: int) -> RolloutBatch:
    ids, mask = stack_sequences(seqs, pad_id(tokenizer), device)
    texts = [tokenizer.decode(s.reshape(-1).tolist(), skip_special_tokens=True) for s in seqs]
    return RolloutBatch(texts, ids, mask, int(num_prompts), int(num_samples))


def trim_continuation(cont: Tensor, eos_token_id: int | None, pad_token_id: int | None) -> Tensor:
    ids = cont.detach().reshape(-1).to(dtype=torch.long).cpu()
    if eos_token_id is not None:
        hits = (ids == eos_token_id).nonzero(as_tuple=False)
        if hits.numel() > 0:
            return ids[: int(hits[0]) + 1]
    if pad_token_id is not None and pad_token_id != eos_token_id:
        hits = (ids == pad_token_id).nonzero(as_tuple=False)
        if hits.numel() > 0:
            return ids[: int(hits[0])]
    return ids


def hf_generate(model: nn.Module, tokenizer: PreTrainedTokenizerBase, prompt_ids: Tensor, prompt_mask: Tensor, *, num_samples: int = 1,
                max_new_tokens: int = 512, temperature: float = 1.0, top_p: float = 1.0,
                seed: int | None = None, do_sample: bool = True) -> RolloutBatch:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if not do_sample and num_samples != 1:
        raise ValueError("greedy generation supports num_samples=1 only")
    model.eval()
    kw = dict(input_ids=prompt_ids, attention_mask=prompt_mask, max_new_tokens=int(max_new_tokens),
              do_sample=do_sample, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if do_sample:
        kw.update(temperature=float(temperature), top_p=float(top_p), num_return_sequences=int(num_samples))
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
    with torch.inference_mode():
        out = model.generate(**kw)
    B, P = int(prompt_ids.shape[0]), int(prompt_ids.shape[1])
    if int(out.shape[0]) != B * num_samples:
        raise RuntimeError(f"generate returned {out.shape[0]} sequences, expected {B * num_samples}")
    seqs = [trim_continuation(out[i, P:], tokenizer.eos_token_id, tokenizer.pad_token_id) for i in range(out.shape[0])]
    return rollout_from_sequences(tokenizer, seqs, device=prompt_ids.device, num_prompts=B, num_samples=num_samples)


def vllm_server_generate(tokenizer: PreTrainedTokenizerBase, prompt_ids: Tensor, prompt_mask: Tensor, *, host: str = "127.0.0.1",
                         port: int, num_samples: int = 1, max_new_tokens: int = 512, temperature: float = 1.0,
                         top_p: float = 1.0, seed: int | None = None, do_sample: bool = True,
                         timeout: float = 1800.0) -> RolloutBatch:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("vllm_server_generate needs the `requests` package") from e
    if not port:
        raise RuntimeError("vllm_server_generate needs the trl vllm-serve HTTP port")
    B = int(prompt_ids.shape[0])
    prompts = [prompt_ids[b][prompt_mask[b].bool()].detach().cpu().tolist() for b in range(B)]
    payload = {
        "prompts": prompts, "images": None, "n": int(num_samples), "repetition_penalty": 1.0,
        "temperature": float(temperature) if do_sample else 0.0, "top_p": float(top_p) if do_sample else 1.0,
        "top_k": 0, "min_p": 0.0, "max_tokens": int(max_new_tokens), "logprobs": 0,
        "structured_outputs_regex": None,
        "generation_kwargs": ({"seed": int(seed)} if seed is not None else {}),
    }
    resp = requests.post(f"http://{host}:{port}/generate/", json=payload, timeout=timeout)
    resp.raise_for_status()
    completion_ids = resp.json()["completion_ids"]
    if len(completion_ids) != B * num_samples:
        raise RuntimeError(f"vLLM server returned {len(completion_ids)} completions, expected {B * num_samples}")
    seqs = [torch.tensor(ids, dtype=torch.long) for ids in completion_ids]
    return rollout_from_sequences(tokenizer, seqs, device=prompt_ids.device, num_prompts=B, num_samples=num_samples)


def generate_rollouts(model: nn.Module, tokenizer: PreTrainedTokenizerBase, prompt_ids: Tensor, prompt_mask: Tensor, *, backend: str = "hf",
                      num_samples: int = 1, max_new_tokens: int = 512, temperature: float = 1.0,
                      top_p: float = 1.0, seed: int | None = None, do_sample: bool = True,
                      server_host: str = "127.0.0.1", server_port: int = 0) -> RolloutBatch:
    if backend == "hf":
        return hf_generate(model, tokenizer, prompt_ids, prompt_mask, num_samples=num_samples,
                           max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed,
                           do_sample=do_sample)
    if backend == "vllm_server":
        return vllm_server_generate(tokenizer, prompt_ids, prompt_mask, host=server_host, port=server_port,
                                    num_samples=num_samples, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_p=top_p, seed=seed, do_sample=do_sample)
    raise ValueError(f"unknown generation backend {backend!r} (expected 'hf' or 'vllm_server')")


__all__ = [
    "RolloutBatch", "rollout_to_completions", "render_prompt", "tokenize_prompt", "tokenize_prompts_batch",
    "pad_id", "stack_sequences", "rollout_from_sequences", "trim_continuation", "hf_generate",
    "vllm_server_generate", "generate_rollouts",
]
