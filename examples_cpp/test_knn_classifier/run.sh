#!/usr/bin/env bash

cd "$(dirname "$0")" &&
"./build/info9-pgm" "../../data/train.hdf5" "../../data/testa.hdf5" 20000 5000 20 128

