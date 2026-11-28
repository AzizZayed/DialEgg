// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_region_opaque.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_region_opaque.mlir --eq-sat --egg-file=scf_region_opaque.egg | FileCheck %s

func.func @f(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = arith.constant 0 : i32
  %two = arith.constant 2 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%s = %s0) -> (i32) {
    %ic = arith.index_cast %i : index to i32
    %t = arith.muli %ic, %two : i32
    vector.print %t : i32
    %u = arith.addi %s, %t : i32
    scf.yield %u : i32
  }
  func.return %r : i32
}

// CHECK-LABEL: func.func @f(%arg0: index) -> i32 {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = scf.for %arg1 = %c0 to %arg0 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
// CHECK-NEXT:     %1 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:     %2 = arith.addi %1, %1 : i32
// CHECK-NEXT:     vector.print %2 : i32
// CHECK-NEXT:     %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:     scf.yield %3 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : i32
// CHECK-NEXT: }
