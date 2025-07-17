#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
fi

# Builds the Dialegg project using CMake.
# Usage: ./build.sh [-d | -r]
# Options:
#   -d     Build in debug mode (default)
#   -r     Build in release mode

# Set default build type
BUILD_TYPE="Debug"

# Parse command line arguments
while getopts "dr" opt; do
    case $opt in
        d)
            BUILD_TYPE="Debug"
            ;;
        r)
            BUILD_TYPE="Release"
            ;;
        *)
            echo "Usage: $0 [-d | -r]"
            exit 1
            ;;
    esac
done

BUILD_DIR="build-$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
LLVM_BUILD=$LLVM_DIR/$BUILD_DIR
echo "Building Dialegg in $BUILD_TYPE mode in directory $BUILD_DIR"

mkdir -p $BUILD_DIR

cmake -S . -B $BUILD_DIR \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir

cmake --build $BUILD_DIR