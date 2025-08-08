// RUN: %eggopt %s --eq-sat | FileCheck %s

func.func @mul2(%x: tensor<4xf64>) -> tensor<4xf64> {
	%cst = stablehlo.constant dense<2.0> : tensor<4xf64>
	%res = stablehlo.multiply %x, %cst : tensor<4xf64>
	return %res : tensor<4xf64>
}

// CHECK: func.func @mul2(%arg0: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf64>
// CHECK-NEXT:    return %0 : tensor<4xf64>
// CHECK-NEXT:  }


func.func @concat(%a: tensor<3x2xf32>, %b: tensor<1x2xf32>, %c: tensor<3x2xf32>, %d: tensor<1x2xf32>) -> tensor<4x2xf32> {
	%ab = stablehlo.concatenate %a, %b, dim = 0 : (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<4x2xf32>
	%cd = stablehlo.concatenate %c, %d, dim = 0 : (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<4x2xf32>
	%res = stablehlo.add %ab, %cd : tensor<4x2xf32>
	return %res : tensor<4x2xf32>
}

// CHECK: func.func @concat(%arg0: tensor<3x2xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<3x2xf32>, %arg3: tensor<1x2xf32>) -> tensor<4x2xf32> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg2 : tensor<3x2xf32>
// CHECK-NEXT:     %1 = stablehlo.add %arg1, %arg3 : tensor<1x2xf32>
// CHECK-NEXT:     %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:     return %2 : tensor<4x2xf32>
// CHECK-NEXT:  }


func.func @_2mm(%x: tensor<100x10xi64>, %y: tensor<10x150xi64>, %z: tensor<150x8xi64>) -> tensor<100x8xi64> {
    // (xy) z cost ac(b+d) = 100*150*(10+8) = 270,000
    // x (yz) cost bd(c+a) = 10*8*(150+100) = 20,000
    
    %xy = stablehlo.dot_general %x, %y, contracting_dims = [1] x [0] : (tensor<100x10xi64>, tensor<10x150xi64>) -> tensor<100x150xi64>
    %xy_z = stablehlo.dot_general %xy, %z, contracting_dims = [1] x [0] : (tensor<100x150xi64>, tensor<150x8xi64>) -> tensor<100x8xi64>

    func.return %xy_z : tensor<100x8xi64>
}

// CHECK: func.func @_2mm(%arg0: tensor<100x10xi64>, %arg1: tensor<10x150xi64>, %arg2: tensor<150x8xi64>) -> tensor<100x8xi64> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0] : (tensor<10x150xi64>, tensor<150x8xi64>) -> tensor<10x8xi64>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<100x10xi64>, tensor<10x8xi64>) -> tensor<100x8xi64>
// CHECK-NEXT:     return %1 : tensor<100x8xi64>
// CHECK-NEXT: }


func.func @_3mm(%x: tensor<200x175xi64>, %y: tensor<175x250xi64>, %z: tensor<250x150xi64>, %w: tensor<150x10xi64>) -> tensor<200x10xi64> {
    %xy = stablehlo.dot_general %x, %y, contracting_dims = [1] x [0] : (tensor<200x175xi64>, tensor<175x250xi64>) -> tensor<200x250xi64>
    %xy_z = stablehlo.dot_general %xy, %z, contracting_dims = [1] x [0] : (tensor<200x250xi64>, tensor<250x150xi64>) -> tensor<200x150xi64>
    %xy_z__w = stablehlo.dot_general %xy_z, %w, contracting_dims = [1] x [0] : (tensor<200x150xi64>, tensor<150x10xi64>) -> tensor<200x10xi64>

    func.return %xy_z__w : tensor<200x10xi64>
}

// CHECK: func.func @_3mm(%arg0: tensor<200x175xi64>, %arg1: tensor<175x250xi64>, %arg2: tensor<250x150xi64>, %arg3: tensor<150x10xi64>) -> tensor<200x10xi64> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] : (tensor<250x150xi64>, tensor<150x10xi64>) -> tensor<250x10xi64>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0] : (tensor<175x250xi64>, tensor<250x10xi64>) -> tensor<175x10xi64>
// CHECK-NEXT:     %2 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0] : (tensor<200x175xi64>, tensor<175x10xi64>) -> tensor<200x10xi64>
// CHECK-NEXT:     return %2 : tensor<200x10xi64>
// CHECK-NEXT: }

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