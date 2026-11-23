// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_multi_if.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_multi_if.mlir --eq-sat --egg-file=scf_multi_if.egg | FileCheck %s

func.func @g(%c: i1, %x: i32) -> (i32, i32) {
  %one = arith.constant 1 : i32
  %r:2 = scf.if %c -> (i32, i32) {
    %a = arith.addi %x, %one : i32
    %two = arith.constant 2 : i32
    %b = arith.muli %x, %two : i32
    scf.yield %a, %b : i32, i32
  } else {
    %d = arith.subi %x, %one : i32
    scf.yield %d, %x : i32, i32
  }
  func.return %r#0, %r#1 : i32, i32
}

// CHECK-LABEL: func.func @g(%arg0: i1, %arg1: i32) -> (i32, i32) {
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %c1_i32_0 = arith.constant 1 : i32
// CHECK-NEXT:   %0:2 = scf.if %arg0 -> (i32, i32) {
// CHECK-NEXT:     %1 = arith.addi %arg1, %c1_i32 : i32
// CHECK-NEXT:     %2 = arith.addi %arg1, %arg1 : i32
// CHECK-NEXT:     scf.yield %1, %2 : i32, i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %1 = arith.subi %arg1, %c1_i32_0 : i32
// CHECK-NEXT:     scf.yield %1, %arg1 : i32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#0, %0#1 : i32, i32
// CHECK-NEXT: }
