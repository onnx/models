# Checks whether a slurm output contains any errors
SLURM_OUTPUT="$1"
if ! grep -q "Model successfully built!" $SLURM_OUTPUT
then
    cat $SLURM_OUTPUT
    echo "Model has not been successfully built"
    exit 1
fi
if grep -q "CommandNotFoundError" $SLURM_OUTPUT
then
    cat $SLURM_OUTPUT
    echo "CommandNotFoundError fount in slurm output. This is likely due to conda not being found."
    exit 1
fi
