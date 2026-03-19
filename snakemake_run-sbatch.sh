#!/bin/bash
#SBATCH --job-name=BeeGees_pipeline
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=snakemake_run_%j.out
#SBATCH --error=snakemake_run_%j.err

## Conda environment
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
 
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: Cannot find conda. Please ensure conda is on your PATH or set CONDA_BASE."
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
 
conda activate BeeGees_env


# Setup logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VERSION="v2.0.0"
PIPELINE="BeeGees Snakemake workflow"				
LOG_FILE="snakemake_${TIMESTAMP}.log"
CONFIG="./config/config.yaml"
PROFILE="./profiles/slurm/"

# Function for timestamped logging
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log_with_timestamp "=== Snakemake Workflow Start ==="
log_with_timestamp "Pipeline: $PIPELINE"
log_with_timestamp "Version: $VERSION"
log_with_timestamp "Log file: $LOG_FILE"
log_with_timestamp "Running on: $(hostname)"
log_with_timestamp "Screen session: $STY"
log_with_timestamp "Working directory: $(pwd)"
log_with_timestamp "Conda environment: $CONDA_DEFAULT_ENV"

##Snakemake
# Unlock directory
log_with_timestamp "Unlocking Snakemake directory..."
snakemake --profile "$PROFILE" \
    --snakefile ./workflow/Snakefile \
	--configfile "$CONFIG" \
	--unlock

# Run snakemake workflow with profile
log_with_timestamp "Starting workflow execution..."
snakemake --profile "$PROFILE" \
    --snakefile ./workflow/Snakefile \
    --configfile "$CONFIG" \
    --rerun-incomplete \
    2>&1 | tee -a "$LOG_FILE"

log_with_timestamp "Log file saved as: $LOG_FILE"
