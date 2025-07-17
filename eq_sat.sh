#!/bin/bash

# Usage
# ./eq_sat.sh [-v] <input_file> [-e <egg_file>] [-o <output_file>]
#   -v               : verbose mode
#   <input_file>     : the input MLIR file to process
#   -e <egg_file>    : egg file to use
#   -o <output_file> : optional output file to save the result

VERBOSE=false
INPUT_FILE=""
EGG_FILE=""
OUTPUT_FILE=""
BUILD_TYPE="Debug"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v)
            VERBOSE=true
            shift
            ;;
        -e)
            EGG_FILE="$2"
            shift 2
            ;;
        -o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -d)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r)
            BUILD_TYPE="Release"
            shift
            ;;
        *)
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
            fi
            shift
            ;;
    esac
done

BUILD_DIR="build-$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"

#### Validation
if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: Input file is required."
    echo "Usage: $0 [-v] <input_file> [-e <egg_file>] [-o <output_file>]"
    exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi
if [[ -n "$EGG_FILE" && ! -f "$EGG_FILE" ]]; then
    echo "Error: Egg file '$EGG_FILE' does not exist."
    exit 1
fi
if [[ -n "$OUTPUT_FILE" && -d "$OUTPUT_FILE" ]]; then
    echo "Error: Output file '$OUTPUT_FILE' is a directory, not a file."
    exit 1
fi

#### Log info
echo "Running equality saturation on file: $INPUT_FILE $(if "$VERBOSE"; then echo 'in verbose mode'; fi)"
if [[ -n "$EGG_FILE" ]]; then
    echo "Using egg file: $EGG_FILE"
fi
if [[ -n "$OUTPUT_FILE" ]]; then
    echo "Output will be saved to: $OUTPUT_FILE"
fi

# If verbose, add "--debug-only=dialegg", otherwise add nothing
# if there is an egg file, add "--egg-file=<egg_file>", otherwise add nothing
# if there is an output file, add "-o <output_file>", otherwise add nothing

./$BUILD_DIR/egg-opt --eq-sat --canonicalize --cse "$INPUT_FILE" \
    $(if $VERBOSE; then echo "--debug-only=dialegg"; fi) \
    $(if [[ -n "$EGG_FILE" ]]; then echo "--egg-file=$EGG_FILE"; fi) \
    $(if [[ -n "$OUTPUT_FILE" ]]; then echo "-o $OUTPUT_FILE"; fi)
