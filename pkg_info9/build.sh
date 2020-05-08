#!/usr/bin/env bash

# Build the c++ extension using CMake
# This script will not install the extension.
# To install the extension (and build it if it not already) run
#     python -m pip install .
# in this directory.




cmake -S . -B "./build" -GNinja && cmake --build "./build"


# -pipe
#        -Wp,-D_FORTIFY_SOURCE=2  -Wp,-D_GLIBCXX_ASSERTIONS
#        -fexceptions -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
#        -fPIE -Wl,-z,noexecstack,-z,relro,-z,defs,-pie,-z,now
#        PRIVATE -Wall -Wextra -Wpedantic -Wformat=2 -Wswitch-default -Wswitch-enum -Wfloat-equal -Wno-conversion
#        -pedantic-errors -Werror=format-security
