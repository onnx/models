#!/usr/bin/env bash
SCRIPT="$1"
ARGS="$2"
WORKING_DIRECTORY="$3"
ML_CACHE="$4"

source "$(dirname $(dirname $(which conda)))"/etc/profile.d/conda.sh
conda activate tracker_slurm
export USING_SLURM="TRUE"

# Export ML cache dir if needed
if [ "$ML_CACHE" ]
then
  export HF_DATASETS_CACHE="$ML_CACHE""/huggingface"
  export TRANSFORMERS_CACHE="$ML_CACHE""/huggingface"
  export HF_HOME="$ML_CACHE""/huggingface"
  export XDG_CACHE_HOME="$ML_CACHE""/huggingface"
  export TORCH_HOME="$ML_CACHE""/torch-hub"
  export TORCH_HUB="$ML_CACHE""/torch-hub"
fi
umask 002
TKML_PATH=$(python -c "import turnkeyml; print(turnkeyml.__path__[0])")
python "$TKML_PATH"/cli/login.py --key "${HUGGINGFACE_API_KEY}"
cd "$WORKING_DIRECTORY" || exit
# shellcheck disable=SC2116,SC2046
"$SCRIPT" $(echo "$ARGS")
conda deactivate
