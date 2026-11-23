// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_multi_for.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_multi_for.mlir --eq-sat --egg-file=scf_multi_for.egg | FileCheck %s

func.func @f(%n: index) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = arith.constant 0 : i32
  %p0 = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%s = %s0, %p = %p0) -> (i32, i32) {
    %s1 = arith.addi %s, %two : i32
    %m = arith.muli %p, %two : i32
    scf.yield %s1, %m : i32, i32
  }
  func.return %r#0, %r#1 : i32, i32
}

// CHECK-LABEL: func.func @f(%arg0: index) -> (i32, i32) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:   %0:2 = scf.for %arg1 = %c0 to %arg0 step %c1 iter_args(%arg2 = %c0_i32, %arg3 = %c1_i32) -> (i32, i32) {
// CHECK-NEXT:     %1 = arith.addi %arg2, %c2_i32 : i32
// CHECK-NEXT:     %2 = arith.addi %arg3, %arg3 : i32
// CHECK-NEXT:     scf.yield %1, %2 : i32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#0, %0#1 : i32, i32
// CHECK-NEXT: }
