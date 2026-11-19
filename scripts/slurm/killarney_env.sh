# source me inside a job on Killarney (Alliance/Vector): modules first, then the venv; login nodes have internet, compute nodes may not.
module load StdEnv/2023 python/3.11.5 cuda/12.6 arrow/25.0.0
source "${VENV_DIR:-$HOME/envs/influence-rlvr}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
