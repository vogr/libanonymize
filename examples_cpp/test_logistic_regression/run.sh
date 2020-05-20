#!/usr/bin/env bash

cd "$(dirname "$0")" &&
"./build/logistic" "../../data/train.hdf5" "../../data/testa.hdf5" 0 0 10 "RMSProp" "None"

