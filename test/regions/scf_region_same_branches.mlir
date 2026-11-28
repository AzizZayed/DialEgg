// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_region_same_branches.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_region_same_branches.mlir --eq-sat --egg-file=scf_region_same_branches.egg | FileCheck %s

func.func @f(%c: i1, %x: i32) -> i32 {
  %two = arith.constant 2 : i32
  %r = scf.if %c -> (i32) {
    %a = arith.muli %x, %two : i32
    scf.yield %a : i32
  } else {
    %b = arith.muli %x, %two : i32
    scf.yield %b : i32
  }
  func.return %r : i32
}

// CHECK-LABEL: func.func @f(%arg0: i1, %arg1: i32) -> i32 {
// CHECK-NEXT:   %0 = scf.if %arg0 -> (i32) {
// CHECK-NEXT:     %1 = arith.addi %arg1, %arg1 : i32
// CHECK-NEXT:     scf.yield %1 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %1 = arith.addi %arg1, %arg1 : i32
// CHECK-NEXT:     scf.yield %1 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : i32
// CHECK-NEXT: }
