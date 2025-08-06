// RUN: %eggopt %s --eq-sat | FileCheck %s

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
