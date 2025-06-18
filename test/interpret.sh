#!/bin/bash

# Script to interpret an MLIR file with lli.

# Usage: ./test/interpret.sh <file>
MLIR_FILE="$1"
MLIR_FILEPATH=$(dirname $MLIR_FILE) # directory of the MLIR file
MLIR_FILENAME=$(basename $MLIR_FILE .mlir) # name without path or extension

echo "Interpreting $MLIR_FILE"

MLIR_LL_FILE="$MLIR_FILEPATH/$MLIR_FILENAME.ll.mlir"
./build/egg-opt --stablehlo-legalize-to-linalg \
         --convert-elementwise-to-linalg \
         --convert-tensor-to-linalg \
         --convert-linalg-to-loops \
         --one-shot-bufferize=bufferize-function-boundaries=true \
         --convert-linalg-to-loops \
         --expand-strided-metadata \
         --lower-affine \
         --convert-index-to-llvm \
         --convert-math-to-llvm \
         --convert-scf-to-cf \
         --convert-cf-to-llvm \
         --convert-arith-to-llvm \
         --convert-func-to-llvm \
         --finalize-memref-to-llvm \
         --reconcile-unrealized-casts \
         "$MLIR_FILE" -o "$MLIR_LL_FILE"

LL_FILE="$MLIR_FILEPATH/$MLIR_FILENAME.ll"
mlir-translate --mlir-to-llvmir "$MLIR_LL_FILE" -o "$LL_FILE"

lli --dlopen="/Users/aziz/dev/lib/llvm/build-debug/lib/libmlir_c_runner_utils.dylib" --dlopen="/Users/aziz/dev/current/dialegg/test/util/libutil.dylib" "$LL_FILE"
# TODO remove hardcoded paths