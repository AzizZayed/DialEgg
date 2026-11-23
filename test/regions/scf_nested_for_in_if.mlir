// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_nested_for_in_if.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_nested_for_in_if.mlir --eq-sat --egg-file=scf_nested_for_in_if.egg | FileCheck %s

func.func @f(%n: index, %c: i1, %x: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %two = arith.constant 2 : i32
  %r = scf.if %c -> (i32) {
    %l = scf.for %i = %c0 to %n step %c1 iter_args(%s = %x) -> (i32) {
      %m = arith.muli %s, %two : i32
      scf.yield %m : i32
    }
    scf.yield %l : i32
  } else {
    scf.yield %x : i32
  }
  func.return %r : i32
}

// CHECK-LABEL: func.func @f(%arg0: index, %arg1: i1, %arg2: i32) -> i32 {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = scf.if %arg1 -> (i32) {
// CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %arg0 step %c1 iter_args(%arg4 = %arg2) -> (i32) {
// CHECK-NEXT:       %2 = arith.addi %arg4, %arg4 : i32
// CHECK-NEXT:       scf.yield %2 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %1 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %arg2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : i32
// CHECK-NEXT: }
