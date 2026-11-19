"""Every number printed in the paper, as printed (toy Table 1 and its p-values, Table 4, Table 6, Table 7, Table 8).
The analysis scripts never read this file; `python -m experiments.compare_to_paper <table.json>` does."""
from __future__ import annotations

TABLE1 = {
    "eta_over_100": {"dot": 0.906, "adam": 0.995, "ekfac": 0.922, "cg": 0.873},
    "eta_over_10": {"dot": 0.821, "adam": 0.930, "ekfac": 0.922, "cg": 0.894},
    "eta": {"dot": 0.523, "adam": 0.645, "ekfac": 0.699, "cg": 0.717},
    "p2": {"dot": 0.503, "adam": 0.708, "ekfac": 0.751, "cg": 0.794},
}
TABLE1_P_VALUES = {"eta_over_10": {("adam", "ekfac"): 0.04}}
TABLE1_P_MAX_OTHERS = 0.01

TABLE4 = {
    "mlp/1": {"D": 421, "spearman": 0.98, "log_pearson": 0.98, "rel_err_pct": 34},
    "mlp/8": {"D": 421, "spearman": 0.98, "log_pearson": 0.98, "rel_err_pct": 29},
    "mlp/32": {"D": 421, "spearman": 0.98, "log_pearson": 0.98, "rel_err_pct": 27},
    "lm": {"D": 768, "spearman": 0.99, "log_pearson": 0.97, "rel_err_pct": 28},
    "qwen_policy_fisher": {"D": 3.7e7, "spearman": 0.99, "log_pearson": 1.00, "rel_err_pct": 78},
    "qwen_empirical_fisher": {"D": 3.7e7, "spearman": 0.98, "log_pearson": 0.99, "rel_err_pct": 41},
}

TABLE6 = {
    100: {1: {"dot": .18, "adam": .11, "ekfac": .02, "control": .12},
          50: {"dot": .03, "adam": .05, "ekfac": -.10, "control": .07},
          100: {"dot": -.05, "adam": -.04, "ekfac": -.14, "control": .21},
          150: {"dot": -.02, "adam": .00, "ekfac": -.09, "control": .18},
          200: {"dot": -.19, "adam": -.08, "ekfac": -.19, "control": .23}},
    200: {1: {"dot": .12, "adam": .14, "ekfac": -.00, "control": -.02},
          50: {"dot": .03, "adam": -.00, "ekfac": .00, "control": .15},
          100: {"dot": .03, "adam": .04, "ekfac": .11, "control": .00},
          150: {"dot": .07, "adam": -.00, "ekfac": -.09, "control": .14},
          200: {"dot": .10, "adam": .17, "ekfac": .02, "control": -.02}},
}

TABLE7_COLUMNS = ("observed_sd", "noise_floor", "residual_sd", "reliability", "ci_lo", "ci_hi")
TABLE7 = {
    100: {1: (.0141, .0195, .0000, -.07, -.23, .10), 50: (.0207, .0183, .0096, .19, .06, .30),
          100: (.0188, .0182, .0046, .11, -.04, .24), 150: (.0173, .0180, .0000, -.05, -.20, .11),
          200: (.0186, .0179, .0051, .08, -.05, .22)},
    200: {1: (.0159, .0193, .0000, .05, -.09, .20), 50: (.0180, .0181, .0000, .09, -.05, .23),
          100: (.0176, .0178, .0000, .01, -.12, .13), 150: (.0177, .0177, .0000, .07, -.09, .20),
          200: (.0189, .0176, .0067, .08, -.08, .21)},
}

TABLE8_COLUMNS = ("auc", "precision_at_k", "poison_in_top_20")
TABLE8 = {
    "single_step1": (0.865, 0.66, 1.00),
    "single_step101": (0.908, 0.71, 1.00),
    "single_step199": (0.956, 0.80, 1.00),
    "visited_offset0": (0.824, 0.58, 0.95),
    "visited_offset12": (0.832, 0.59, 0.95),
    "momentum_offset0": (0.826, 0.58, 0.95),
    "momentum_offset12": (0.833, 0.60, 0.95),
    "sum_all": (0.923, 0.72, 1.00),
    "window_151_200": (0.951, 0.78, 1.00),
    "rollout_sum": (0.569, 0.37, None),
    "chance": (0.500, 0.20, 0.20),
}

TOLERANCE = {
    "table1": 0.0015,
    "table4": {"D": 0.05, "spearman": 0.005, "log_pearson": 0.005, "rel_err_pct": 0.5},
    "table6": 0.0055,
    "table7": {"observed_sd": 5e-5, "noise_floor": 5e-5, "residual_sd": 5e-5,
               "reliability": 0.03, "ci_lo": 0.06, "ci_hi": 0.06},
    "table8": {"auc": 0.0015, "precision_at_k": 0.006, "poison_in_top_20": 0.006},
}
