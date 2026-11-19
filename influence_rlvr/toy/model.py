from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

Tensor = torch.Tensor

ALL_SEQUENCES = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)


class ToyAutoregressiveMLP(nn.Module):
    def __init__(self, input_dim: int = 65, hidden_dim: int = 16, output_dim: int = 2, use_bias: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.first = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=use_bias),
        )
        self.second = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=use_bias),
        )

    def first_token_logits(self, z: Tensor) -> Tensor:
        return self.first(z)

    def second_token_logits(self, z: Tensor, first_tokens: Tensor) -> Tensor:
        if first_tokens.ndim == 1:
            first_tokens = first_tokens.unsqueeze(1)
        return self.second(torch.cat([z, first_tokens.to(dtype=z.dtype)], dim=1))

    def per_token_log_probs(self, z: Tensor, sequences: Tensor) -> Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.to(dtype=torch.float32)
        z_rep = z.expand(int(sequences.shape[0]), -1)
        first_tokens = sequences[:, 0]
        first_lp = F.log_softmax(self.first_token_logits(z_rep), dim=-1).gather(1, first_tokens.unsqueeze(1)).squeeze(1)
        second_lp = F.log_softmax(self.second_token_logits(z_rep, first_tokens.float()), dim=-1).gather(
            1, sequences[:, 1].unsqueeze(1)).squeeze(1)
        return torch.stack([first_lp, second_lp], dim=1)

    def sequence_log_probs(self, z: Tensor, sequences: Tensor) -> Tensor:
        return self.per_token_log_probs(z, sequences).sum(dim=1)

    def exact_sequence_distribution(self, z: Tensor) -> tuple[Tensor, Tensor]:
        sequences = ALL_SEQUENCES.to(self.device)
        return sequences, self.sequence_log_probs(z.to(self.device), sequences).exp()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def init_reference(input_dim: int, hidden_dim: int, seed: int) -> ToyAutoregressiveMLP:
    torch.manual_seed(seed)
    model = ToyAutoregressiveMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model
