// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_nested_for_in_for.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_nested_for_in_for.mlir --eq-sat --egg-file=scf_nested_for_in_for.egg | FileCheck %s

func.func @f(%n: index, %x: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %two = arith.constant 2 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%s = %x) -> (i32) {
    %q = scf.for %j = %c0 to %n step %c1 iter_args(%t = %s) -> (i32) {
      %m = arith.muli %t, %two : i32
      scf.yield %m : i32
    }
    scf.yield %q : i32
  }
  func.return %r : i32
}

// CHECK-LABEL: func.func @f(%arg0: index, %arg1: i32) -> i32 {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = scf.for %arg2 = %c0 to %arg0 step %c1 iter_args(%arg3 = %arg1) -> (i32) {
// CHECK-NEXT:     %1 = scf.for %arg4 = %c0 to %arg0 step %c1 iter_args(%arg5 = %arg3) -> (i32) {
// CHECK-NEXT:       %2 = arith.addi %arg5, %arg5 : i32
// CHECK-NEXT:       scf.yield %2 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %1 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : i32
// CHECK-NEXT: }
