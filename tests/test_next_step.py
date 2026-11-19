from __future__ import annotations

import torch
from torch import nn

from experiments.llm.next_step import adamw_step, adamw_trajectory
from influence_rlvr.preconditioner import load_adam_state
from influence_rlvr.utils import grad_vector, trainable_params


def test_closed_form_step_equals_torch_adamw_with_clipping(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1)
    x, y = torch.randn(16, 5), torch.randn(16, 2)
    for _ in range(3):
        opt.zero_grad()
        (10 * (model(x) - y).pow(2).mean()).backward()
        opt.step()
    torch.save(opt.state_dict(), tmp_path / "optimizer.pt")
    state = load_adam_state(model, tmp_path)
    theta = torch.cat([p.detach().reshape(-1) for p in trainable_params(model)])
    loss = 10 * (model(x) - y).pow(2).mean()
    predicted = adamw_step(theta, grad_vector(model, loss, retain_graph=True), state, max_grad_norm=1.0)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    actual = torch.cat([p.detach().reshape(-1) for p in trainable_params(model)])
    assert state["t"] == 3 and torch.allclose(predicted, actual, atol=1e-7)


def test_frozen_gradient_trajectory_equals_repeated_adamw_steps(tmp_path):
    torch.manual_seed(1)
    model = nn.Linear(6, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1)
    for _ in range(2):
        opt.zero_grad()
        model(torch.randn(8, 6)).pow(2).mean().backward()
        opt.step()
    torch.save(opt.state_dict(), tmp_path / "optimizer.pt")
    state = load_adam_state(model, tmp_path)
    params = trainable_params(model)
    theta = torch.cat([p.detach().reshape(-1) for p in params])
    g = [0.01 * torch.randn_like(p) for p in params]
    out = dict(zip([1, 4], adamw_trajectory(theta, torch.cat([x.reshape(-1) for x in g]), state, [1, 4], max_grad_norm=0.0)))
    for k in range(1, 5):
        for p, x in zip(params, g):
            p.grad = x.clone()
        opt.step()
        if k in out:
            assert torch.allclose(out[k], torch.cat([p.detach().reshape(-1) for p in params]), atol=1e-7)
