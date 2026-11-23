// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_while.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_while.mlir --eq-sat --egg-file=scf_while.egg | FileCheck %s

func.func @f(%n: i32, %x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %r:2 = scf.while (%i = %c0, %s = %x) : (i32, i32) -> (i32, i32) {
    %c = arith.cmpi slt, %i, %n : i32
    scf.condition(%c) %i, %s : i32, i32
  } do {
  ^bb0(%j: i32, %t: i32):
    %m = arith.muli %t, %two : i32
    %j1 = arith.addi %j, %c1 : i32
    scf.yield %j1, %m : i32, i32
  }
  func.return %r#1 : i32
}

// CHECK-LABEL: func.func @f(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %0:2 = scf.while (%arg2 = %c0_i32, %arg3 = %arg1) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:     %1 = arith.cmpi slt, %arg2, %arg0 : i32
// CHECK-NEXT:     scf.condition(%1) %arg2, %arg3 : i32, i32
// CHECK-NEXT:   } do {
// CHECK-NEXT:   ^bb0(%arg2: i32, %arg3: i32):
// CHECK-NEXT:     %1 = arith.addi %arg3, %arg3 : i32
// CHECK-NEXT:     %2 = arith.addi %arg2, %c1_i32 : i32
// CHECK-NEXT:     scf.yield %2, %1 : i32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : i32
// CHECK-NEXT: }
