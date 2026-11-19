import numpy as np

from experiments.llm.select.grad_profile import energy_profile, preconditioner_stats, rank_agreement


def test_block_scalar_preconditioner_cannot_separate_dot_from_adam():
    rng = np.random.default_rng(0)
    segs = [("a", 0, 50), ("b", 50, 200)]
    P = np.concatenate([np.full(50, 10.0), np.full(150, 0.1)])  # all spread between tensors
    st = preconditioner_stats(P, segs)
    assert st["between_tensor_variance_fraction"] > 0.999 and st["median_within_tensor_99_1"] == 1.0
    G, h = rng.normal(size=(40, 200)), rng.normal(size=200)
    Pblock = np.concatenate([np.full(b - a, P[a:b].mean()) for _, a, b in segs])
    r = rank_agreement({"dot": G @ h, "adam": G @ (P * h), "block": G @ (Pblock * h)}, np.ones(40, bool))
    assert r["spearman_block_adam"] == 1.0  # block-scalar == adam exactly here
    E = energy_profile(G, segs)
    assert np.allclose(E.sum(1), 1) and E.shape == (40, 2)
