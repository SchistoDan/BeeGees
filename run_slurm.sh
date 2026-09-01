#!/bin/bash
#SBATCH --job-name=BeeGees_pipeline
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=beegees_run_%j.out
#SBATCH --error=beegees_run_%j.err

## Run BeeGees pipeline via SLURM (each rule submitted as a separate SLURM job).
## Activate the beegees conda environment before submitting: sbatch run_slurm.sh

beegees run \
    --config beegees/config/config.yaml \
    --profile slurm \
    --log-file beegees.log \
    2>&1
