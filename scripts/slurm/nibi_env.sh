# source me inside a job on Nibi/Alliance: modules first, then the venv; compute nodes are offline.
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/21.0.0 opencv/4.13.0
source "${VENV_DIR:-$HOME/envs/influence-rlvr}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
