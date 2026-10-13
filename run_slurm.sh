#!/bin/bash
#SBATCH --job-name=BeeGees_pipeline
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=beegees_run_%j.out
#SBATCH --error=beegees_run_%j.err

## Run BeeGees pipeline via SLURM (each rule submitted as a separate SLURM job).
## Submit from a directory initialised with `beegees init`, with the conda
## environment BeeGees is installed into activated: sbatch run_slurm.sh

beegees run \
    --config config/config.yaml \
    --profile slurm \
    --log-file beegees.log \
    2>&1
