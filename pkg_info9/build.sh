#!/usr/bin/env bash

# Build the c++ extension using CMake.
# This script will not install the extension.
# To install the extension (and build it if it not already) run
#     python -m pip install .
# in this directory.



cd "$(dirname "$0")" &&
cmake -S . -B "./build" -GNinja -DCMAKE_BUILD_TYPE=${1:Debug} &&
cmake --build "./build"

