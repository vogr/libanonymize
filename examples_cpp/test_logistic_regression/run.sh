#!/usr/bin/env bash

cd "$(dirname "$0")" &&
"./build/logistic" "../../data/train.hdf5" "../../data/testa.hdf5" 50000 150000 10 "Newton"

