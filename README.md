# Influence Functions in Reinforcement Learning with Verifiable Rewards

Code for the MSc thesis *Influence Functions in Reinforcement Learning with Verifiable Rewards* (Thai Ha Bui, Delft
University of Technology, 2026). It scores how much a training prompt changes a target behaviour of a model trained
with GRPO, using one local influence score with three geometries: the gradient dot product, the Adam diagonal and the
damped Fisher (EK-FAC, or CG in the toy). The score is tested against exact counterfactuals in a controlled GRPO
environment and at LLM scale (Qwen2.5-1.5B/3B/7B-Instruct), and used for online data selection and backdoor detection.

## Where each thesis result comes from

| Thesis (Chapter 4) | Code | Shipped artifacts? |
|---|---|---|
| 4.3 Adam's second moment vs the Fisher diagonal (Table 4.3) | `experiments/vhat/qwen_grpo.py`, `table4.py` | yes: `artifacts/vhat/` |
| 4.4 Controlled GRPO environment (Table 4.4, Figures 4.2, 4.3) | `experiments/toy/`, `scripts/toy_all.sh` | yes: `artifacts/toy/` |
| 4.5.2 Protocol 1, one step (Table 4.6, Figure 4.5) | `experiments/llm/next_step.py`, `next_step_report.py`, `variance_sweep.py`, `variance_report.py` | no |
| 4.5.2 Horizon ICE(m) (Figure 4.4) | `experiments/llm/horizon.py`, `scripts/slurm/submit_horizon_3b.sh` | no |
| 4.5.3 Dot vs Adam agreement (Table 4.7) | `experiments/llm/select/dot_vs_adam.py`, `grad_profile.py` | no |
| 4.5.3 Selection, hard maths (Tables 4.9, 4.10, Figure 4.7) | `experiments/llm/select/train.py`, pools from `experiments/llm/curriculum/`, `scripts/slurm/submit_c1*.sh` | no |
| 4.5.3 Selection, maths + safety (Table 4.11) | `select/train.py` with the pool of `experiments/llm/safety/make_pool.py` | no |
| 4.5.3 Selection, faulty verifier (Figure 4.8) | `select/train.py`, pools from `experiments/llm/faulty/`, `scripts/slurm/submit_f2*.sh` | no |
| 4.5.4 Protocol 2, whole-run retraining (Table 4.12) | `experiments/llm/p2/retrain.py`, `lds.py`, `scripts/slurm/submit_p2.sh` | no |
| 4.5.5 Sandbagging backdoor (Tables 4.13, 4.14, Figures 4.9, 4.10) | `experiments/llm/backdoor/`, `scripts/llm_backdoor.sh` | yes: `artifacts/llm/backdoor/` |
| LLM chapter figures | `experiments/thesis_figures.py` (reads `outputs/llm/`) | no |

"Shipped artifacts" means the table can be re-derived on a CPU from `artifacts/` without rerunning the GPU stages.
The other LLM results need the runs. Selection runs are summarized with `experiments/llm/select/analyze.py`, per-problem
final evals with `eval_final.py`, and paired tests between arms with `paired.py`.

## Layout

```
influence_rlvr/
  attribution/      the estimator core shared by the toy and the LLM
    base.py         Estimator (solve h = M g_test, influence = h @ g_train^T), influence_matrix (stacked or streamed, cosine)
    dot.py adam.py cg.py ekfac.py      M = I | lr·diag(1/(sqrt(v_hat)+eps)) | (F+λI)^-1 by CG | (F_ekfac+λI)^-1
    fvp.py          Fisher-vector products: exact gradient cache (toy), matrix-free per-token policy Fisher (LLM)
    trajectory.py   Protocol 2 sums (all / visited checkpoints / momentum tail), visit-weighted subset predictors
    metrics.py      LDS (Spearman), bootstrap bands and paired p-values, split-half reliability, AUC / precision@k
  preconditioner.py Adam v_hat and P from a live optimizer or a checkpoint's optimizer.pt
  toy/              2-token autoregressive MLP, clustered fixture, exhaustive-rollout GRPO, gradients, Fisher cache
  llm/              Qwen + LoRA loading, prompts and rewards, HF / vLLM sampling, GRPO and gold gradients,
                    EK-FAC / FVP for LoRA, streaming pool scorer, selection step (select.py), pass-rate eval
experiments/
  toy/              Protocol 1 and 2 in the controlled environment, Table 4.4, Figures 4.2-4.3
  vhat/             v_hat against the policy Fisher and the GRPO second moment
  llm/next_step.py horizon.py variance_sweep.py     Protocol 1 at LLM scale
  llm/select/       selection-GRPO driver and its analyses
  llm/curriculum/ faulty/ safety/ hint/             pool builders (hint is not in the thesis)
  llm/p2/           Protocol 2 retraining and LDS
  llm/backdoor/     sandbagging backdoor: data, training, gold / rollout scoring, tables, ROC
  llm/continuation/ an earlier LLM LDS experiment on GSM8K subsets (not in the thesis)
scripts/            one driver per experiment, Slurm templates in scripts/slurm/
data/llm/           shipped inputs: pools, guides, targets, eval sets, backdoor candidates
artifacts/          stored outputs that the analysis scripts re-derive tables from
tests/              pytest suite (CPU)
```

Every estimator is `Estimator.solve(g_test) -> h` followed by `h @ g_train^T`, built through
`influence_rlvr.attribution.make_estimator("dot" | "adam" | "ekfac" | "cg", ...)`. The toy passes stacked fp32
gradients, the LLM streams LoRA gradients in chunks.

Sign convention: `g_train = ∇L_GRPO` (the loss the optimizer descends) and `g_test = ∇(−f)`. The score
`g_test^T M g_train` is then positive when training on the prompt is predicted to increase `f`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[llm,dev]"          # toy only: pip install -e ".[dev]"
pip install vllm                      # optional: vLLM rollouts
pytest -q -m "not slow"               # CPU, under a minute
```

`uv.lock` pins the versions used for the thesis. The LLM experiments ran on single H100s with Slurm.
`scripts/slurm/*_env.sh` are the cluster environments used (module loads, venv, offline Hugging Face cache). Set
`ACCOUNT`, `PARTITION` and `CLUSTER_ENV` for your cluster before using the `submit_*.sh` scripts.

## Run

### Controlled GRPO environment (Section 4.4), about 35 min on a laptop CPU

```bash
bash scripts/toy_all.sh                         # everything into outputs/toy/
bash scripts/toy_smoke.sh                       # 30 s end-to-end smoke test of the same chain
python -m experiments.toy.table1 --artifacts    # Table 4.4 and the p-values from the stored runs
python -m experiments.toy.plots fig2 --artifacts
python -m experiments.toy.plots fig4 --artifacts
```

### Adam's second moment vs the Fisher diagonal (Section 4.3)

```bash
bash scripts/vhat_fisher.sh                     # the Qwen GRPO checkpoint stage needs a GPU
python -m experiments.vhat.table4 --artifacts
```

### Sandbagging backdoor (Section 4.5.5), about 30 GPU-hours

```bash
bash scripts/llm_backdoor.sh                    # data, pilot, pool, GRPO run, gold and rollout scoring, tables
python -m experiments.llm.backdoor.aggregate --artifacts --singles 1,100,200   # Table 4.13 from the stored scores
python -m experiments.llm.backdoor.plot_roc     # Figure 4.10
python -m experiments.llm.backdoor.print_examples                            # Figure 4.9
```

### Selection (Section 4.5.3)

Build a pool with the matching `make_pool*.py`, then queue one run per arm and seed as chained 3 h jobs:

```bash
bash scripts/slurm/submit_select.sh <out> <selector> <method> <seed> <pool> [guide] [slots]
bash scripts/slurm/submit_c1.sh                 # hard maths, all arms
MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/slurm/submit_c1_7b.sh
bash scripts/slurm/submit_f2.sh                 # faulty verifier
python -m experiments.llm.select.analyze --help
```

The thesis arms use the selectors `none`, `random`, `influence` (with `--method dot | adam | ekfac`) and `oracle`.
The target-domain and clean-pool arms are `none` on another pool. `anti`, `liveness` and `disagreement` are
implemented but not reported.

The safety pool is not shipped, because its prompts come from gated or non-commercial datasets (WildJailbreak, XSTest,
BeaverTails). `experiments/llm/safety/make_pool.py` rebuilds it after you accept the dataset terms on Hugging Face.

### Protocol 1 and 2 at LLM scale (Sections 4.5.2 and 4.5.4)

```bash
python -m experiments.llm.next_step --help      # ICE(1) / ACE(1) LDS at a checkpoint
python -m experiments.llm.horizon --help        # ICE(m)
python -m experiments.llm.variance_sweep --help # gradient reproducibility against G
bash scripts/slurm/submit_p2.sh                 # master run, fixed and fresh retrains, LDS
```

All scripts are idempotent (finished units are skipped), take the thesis hyper-parameters as defaults, and list them
with `--help`.

## Notes on reproduction

* The toy is deterministic on CPU, and `scripts/toy_all.sh` regenerates Table 4.4 to three decimals.
  `experiments/compare_to_paper.py` checks produced tables against the values in `experiments/paper_values.py`, which
  hold the numbers of an earlier paper version of this work (the toy table is the same as Table 4.4).
* LLM runs depend on sampling and are reproducible only statistically.
* Where the code and an earlier text disagreed, the code that produced the numbers is kept, and the module docstring
  says so. The toy weights its four exhaustive outputs uniformly and has D = 2196. The backdoor trigger is lowercase in
  the data. In the backdoor pool only the 200 poisoned prompts are filtered to be in-band (the base policy passes 3 to 5
  of 8), while the 800 background prompts are drawn from all 2000 candidates (`assemble_pool.py`).

## License

MIT, see [LICENSE](LICENSE). The shipped data are derived from GSM8K, MATH, MATH-500, DeepScaleR, AIME and
OlympiadBench, under their own licences.
