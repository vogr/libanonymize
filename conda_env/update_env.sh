#!/usr/bin/env bash

# Create the Conda environment `info9_projet`, where all the required libraries will be installed.

cd "$(dirname "$0")" &&
conda env update --name "info_projet9" --file "./conda_env.yml"
