// RUN: rm -rf %t && mkdir %t && cp %s %S/scf_mixed_users.egg %t && cp -r %S/../../src %t/src && cd %t && %eggopt scf_mixed_users.mlir --eq-sat --egg-file=scf_mixed_users.egg | FileCheck %s

func.func @f(%x: i32) -> i32 {
  %two = arith.constant 2 : i32
  %m = arith.muli %x, %two : i32
  vector.print %m : i32
  %r = arith.addi %m, %x : i32
  func.return %r : i32
}

// CHECK-LABEL: func.func @f(%arg0: i32) -> i32 {
// CHECK-NEXT:   %0 = arith.addi %arg0, %arg0 : i32
// CHECK-NEXT:   vector.print %0 : i32
// CHECK-NEXT:   %1 = arith.addi %0, %arg0 : i32
// CHECK-NEXT:   return %1 : i32
// CHECK-NEXT: }
