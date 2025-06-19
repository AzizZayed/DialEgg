# DialEgg: MLIR + Equality Saturation
This tool proposes a dialect-agnostic way to apply equality saturation optimizations to LLVM's Multi-Level IR ([MLIR](https://mlir.llvm.org/)).
The equality saturation engine used is [Egglog](https://github.com/egraphs-good/egglog).

## Getting Started

### LLVM
Clone [LLVM](https://github.com/llvm/llvm-project) with commit `7b074fc9362a4a6a5f172dd8936a22fb25eff96b`.

```bash
git clone https://github.com/llvm/llvm-project.git llvm
git checkout 7b074fc9362a4a6a5f172dd8936a22fb25eff96b
```

The build LLVM core and MLIR in Release mode. This may take a while.

```bash
cd llvm
mkdir build-release

cmake -S llvm -B build-release -G Ninja \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_USE_LINKER=lld

cmake --build build-release
```

If you have issues building, follow the [build guide](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm).

### Egglog
Egglog is the equality saturation engine. This [PR](https://github.com/egraphs-good/egglog/pull/355) has a feature DialEgg needs.

```bash
git clone -b cost-action https://github.com/saulshanabrook/egg-smol.git egglog
cd egglog
git checkout b6e1c96ed7335366e90056ea0a24ef425dfbb8fb
```

Then build it in release mode

```bash
cargo build --release
```

### DialEgg
Within the root directory of this repo, build DialEgg:

```bash
mkdir build
cmake -S . -B build -DLLVM_DIR=[path to llvm cmake dir] -DMLIR_DIR=[path to mlir cmake dir]
cmake --build build
```

Optimize an example MLIR file in the `test` directory.

## Classic Equality Saturation Example
MLIR in [test/classic/classic.mlir](test/classic/classic.mlir):
```llvm
func.func @classic(%a: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %mul = arith.muli %a, %c2 : i32
  %div = arith.divsi %mul, %c2 : i32
  
  func.return %div : i32
}
```

Egglog ops and rules in [test/classic/classic.egg](test/classic/classic.egg):
```lisp
(include "src/base.egg")

;;;; arith dialect ;;;;
(function arith_constant (AttrPair Type) Op)
(function arith_muli (Op Op AttrPair Type) Op)
(function arith_divsi (Op Op Type) Op)
(function arith_shli (Op Op AttrPair Type) Op)

;; OPS HERE ;;

;; RULES HERE ;;
(ruleset rules)
(let c1 (arith_constant (NamedAttr "value" (IntegerAttr 1 (I32))) (I32))) ; 1
(let c2 (arith_constant (NamedAttr "value" (IntegerAttr 2 (I32))) (I32))) ; 2

(rewrite (arith_divsi ?x ?x    (I32)) c1 :ruleset rules) ; x / x = 1
(rewrite (arith_muli  ?x c1 ?a (I32)) ?x :ruleset rules) ; x * 1 = x

(rewrite ; x * 2 = x << 1
    (arith_muli ?x c2 ?a (I32)) ; x * 2
    (arith_shli ?x c1 ?a (I32)) ; x << 1
    :ruleset rules
)
(rewrite ; (xy) / z = x (z / y)
    (arith_divsi   (arith_muli  ?x ?y ?a ?t) ?z ?t)
    (arith_muli ?x (arith_divsi ?y ?z    ?t) ?a ?t)
    :ruleset rules
)

(run-schedule (saturate rules))

;; EXTRACTS HERE ;;
```

Run this and the output will be the optimized MLIR.

```bash
./build/egg-opt --eq-sat test/classic/classic.mlir --egg test/classic/classic.egg
```
Result:
```llvm
func.func @classic(%arg0: i32) -> i32 {
    return %arg0 : i32
}
```

## Future Improvements
- Better support for ops with variadic number of operands and attributes
- Better support for ops with optional operands and attributes