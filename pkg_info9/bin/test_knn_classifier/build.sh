#!/usr/bin/env bash

cd "$(dirname "$0")" &&
cmake -S . -B "./build" -GNinja -DCMAKE_BUILD_TYPE=${1:RelWithDebInfo} &&
cmake --build "./build"
