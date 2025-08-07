// RUN: %eggopt %s --eq-sat | FileCheck %s

func.func @concat_rmatmul(%x : tensor<100x10xi64>, %y : tensor<150x10xi64>, %z : tensor<10x200xi64>) -> tensor<250x200xi64> {
    // (x 0 y)z where 0 is horizontal concatenation (axis 0)
    %x0y = stablehlo.concatenate %x, %y, dim = 0 : (tensor<100x10xi64>, tensor<150x10xi64>) -> tensor<250x10xi64>
    %x0y_z = stablehlo.dot_general %x0y, %z, contracting_dims = [1] x [0] : (tensor<250x10xi64>, tensor<10x200xi64>) -> tensor<250x200xi64>

    func.return %x0y_z : tensor<250x200xi64>
}

// CHECK: func.func @concat_rmatmul(%arg0: tensor<100x10xi64>, %arg1: tensor<150x10xi64>, %arg2: tensor<10x200xi64>) -> tensor<250x200xi64> {
// CHECK-NEXT:     %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<100x10xi64>, tensor<150x10xi64>) -> tensor<250x10xi64>
// CHECK-NEXT:     %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] : (tensor<250x10xi64>, tensor<10x200xi64>) -> tensor<250x200xi64>
// CHECK-NEXT:     return %1 : tensor<250x200xi64>
// CHECK-NEXT: }

func.func @rmatmul_concat(%x : tensor<100x10xi64>, %y : tensor<150x10xi64>, %z : tensor<10x200xi64>) -> tensor<250x200xi64> {
    // xz 0 yz where 0 is vertical concatenation (axis 0)
    %xz = stablehlo.dot_general %x, %z, contracting_dims = [1] x [0] : (tensor<100x10xi64>, tensor<10x200xi64>) -> tensor<100x200xi64>
    %yz = stablehlo.dot_general %y, %z, contracting_dims = [1] x [0] : (tensor<150x10xi64>, tensor<10x200xi64>) -> tensor<150x200xi64>
    %xz0yz = stablehlo.concatenate %xz, %yz, dim = 0 : (tensor<100x200xi64>, tensor<150x200xi64>) -> tensor<250x200xi64>

    func.return %xz0yz : tensor<250x200xi64>
}

// CHECK: func.func @rmatmul_concat(%arg0: tensor<100x10xi64>, %arg1: tensor<150x10xi64>, %arg2: tensor<10x200xi64>) -> tensor<250x200xi64> {
// CHECK-NEXT:     %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<100x10xi64>, tensor<150x10xi64>) -> tensor<250x10xi64>
// CHECK-NEXT:     %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] : (tensor<250x10xi64>, tensor<10x200xi64>) -> tensor<250x200xi64>
// CHECK-NEXT:     return %1 : tensor<250x200xi64>
// CHECK-NEXT: }