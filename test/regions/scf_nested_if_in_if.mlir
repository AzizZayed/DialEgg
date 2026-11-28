// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_nested_if_in_if.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_nested_if_in_if.mlir --eq-sat --egg-file=scf_nested_if_in_if.egg | FileCheck %s

func.func @f(%c: i1, %d: i1, %x: i32) -> i32 {
  %two = arith.constant 2 : i32
  %r = scf.if %c -> (i32) {
    %v = scf.if %d -> (i32) {
      %m = arith.muli %x, %two : i32
      scf.yield %m : i32
    } else {
      scf.yield %x : i32
    }
    scf.yield %v : i32
  } else {
    scf.yield %x : i32
  }
  func.return %r : i32
}

// CHECK-LABEL: func.func @f(%arg0: i1, %arg1: i1, %arg2: i32) -> i32 {
// CHECK-NEXT:   %0 = scf.if %arg0 -> (i32) {
// CHECK-NEXT:     %1 = scf.if %arg1 -> (i32) {
// CHECK-NEXT:       %2 = arith.addi %arg2, %arg2 : i32
// CHECK-NEXT:       scf.yield %2 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %arg2 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %1 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %arg2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : i32
// CHECK-NEXT: }
