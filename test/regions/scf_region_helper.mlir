// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_region_helper.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_region_helper.mlir --eq-sat --egg-file=scf_region_helper.egg | FileCheck %s

func.func @f(%n: index, %x: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %four = arith.constant 4 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%s = %x) -> (i32) {
    %m = arith.muli %s, %four : i32
    scf.yield %m : i32
  }
  func.return %r : i32
}

// CHECK-LABEL: func.func @f(%arg0: index, %arg1: i32) -> i32 {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = scf.for %arg2 = %c0 to %arg0 step %c1 iter_args(%arg3 = %arg1) -> (i32) {
// CHECK-NEXT:     %1 = arith.addi %arg3, %arg3 : i32
// CHECK-NEXT:     %2 = arith.addi %1, %1 : i32
// CHECK-NEXT:     scf.yield %2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : i32
// CHECK-NEXT: }
