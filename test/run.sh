#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
fi

# Script to compile and execute an MLIR file with llc.

# Usage: ./test/interpret.sh <file>
MLIR_FILE="$1"
MLIR_FILEPATH=$(dirname $MLIR_FILE) # directory of the MLIR file
MLIR_FILENAME=$(basename $MLIR_FILE .mlir) # name without path or extension

echo "Executing $MLIR_FILE"

MLIR_LL_FILE="$MLIR_FILEPATH/$MLIR_FILENAME.ll.mlir"
./build-debug/egg-opt --stablehlo-legalize-to-linalg \
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

O_FILE="$MLIR_FILEPATH/$MLIR_FILENAME.o"
llc -filetype=obj -O3 "$LL_FILE" -o "$O_FILE"

EXEC_FILE="$MLIR_FILEPATH/$MLIR_FILENAME.exec"
clang -O3 "$O_FILE" -o "$EXEC_FILE" \
    -L"$LLVM_DEBUG_DIR/lib" -lmlir_c_runner_utils -L"test/util" -lutil -Wl,-rpath,"$LLVM_DEBUG_DIR/lib" -Wl,-rpath,"test/util"

"$EXEC_FILE"