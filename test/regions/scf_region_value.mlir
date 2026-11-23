// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_region_value.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_region_value.mlir --eq-sat --egg-file=scf_region_value.egg | FileCheck %s

func.func @h(%c: i1, %x: i32) -> i32 {
  %one = arith.constant 1 : i32
  %r = scf.if %c -> (i32) {
    %a = arith.muli %x, %one : i32
    scf.yield %a : i32
  } else {
    %d = arith.subi %x, %one : i32
    scf.yield %d : i32
  }
  %o = arith.muli %r, %one : i32
  func.return %o : i32
}

// CHECK-LABEL: func.func @h(%arg0: i1, %arg1: i32) -> i32 {
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %0 = scf.if %arg0 -> (i32) {
// CHECK-NEXT:     scf.yield %arg1 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %1 = arith.subi %arg1, %c1_i32 : i32
// CHECK-NEXT:     scf.yield %1 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : i32
// CHECK-NEXT: }
