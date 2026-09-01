#!/bin/bash
## Run BeeGees pipeline locally (all jobs on the current node).
## Activate the beegees conda environment before running this script.
## Usage: bash run_local.sh

beegees run \
    --config beegees/config/config.yaml \
    --profile local \
    --log-file beegees.log \
    2>&1
