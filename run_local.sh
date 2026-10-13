#!/bin/bash
## Run BeeGees pipeline locally (all jobs on the current node).
## Run from a directory initialised with `beegees init`, with the conda
## environment BeeGees is installed into activated.
## Usage: bash run_local.sh

beegees run \
    --config config/config.yaml \
    --profile local \
    --log-file beegees.log \
    2>&1
