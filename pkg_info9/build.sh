#!/usr/bin/env bash

# Build the c++ extension using CMake
# This script will not install the extension.
# To install the extension (and build it if it not already) run
#     python -m pip install .
# in this directory.

cmake -S . -B "./build" -GNinja && cmake --build "./build"
