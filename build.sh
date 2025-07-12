#!/bin/bash

# Builds the Dialegg project using CMake.
mkdir -p build

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_DIR=~/dev/lib/llvm/build-debug/lib/cmake/llvm \
    -DMLIR_DIR=~/dev/lib/llvm/build-debug/lib/cmake/mlir

cmake --build build