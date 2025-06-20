// RUN: %eggopt %s --eq-sat | FileCheck %s

func.func @_2mm(%x: tensor<100x10xi64>, %y: tensor<10x150xi64>, %z: tensor<150x8xi64>) -> tensor<100x8xi64> {
    // xy_z cost ac(b+d) = 100*150*(10+8) = 270,000
    // x_yz cost bd(c+a) = 10*8*(150+100) = 20,000
    %xy_init = tensor.empty() : tensor<100x150xi64>
    %xy = linalg.matmul ins(%x, %y : tensor<100x10xi64>, tensor<10x150xi64>) 
                        outs(%xy_init : tensor<100x150xi64>) -> tensor<100x150xi64>
    
    %xy_z_init = tensor.empty() : tensor<100x8xi64>
    %xy_z = linalg.matmul ins(%xy, %z : tensor<100x150xi64>, tensor<150x8xi64>) 
                          outs(%xy_z_init : tensor<100x8xi64>) -> tensor<100x8xi64>
    
    return %xy_z : tensor<100x8xi64>
}

// CHECK: func.func @_2mm(%arg0: tensor<100x10xi64>, %arg1: tensor<10x150xi64>, %arg2: tensor<150x8xi64>) -> tensor<100x8xi64> {
// CHECK-NEXT:     %0 = tensor.empty() : tensor<10x8xi64>
// CHECK-NEXT:     %1 = linalg.matmul ins(%arg1, %arg2 : tensor<10x150xi64>, tensor<150x8xi64>) outs(%0 : tensor<10x8xi64>) -> tensor<10x8xi64>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<100x8xi64>
// CHECK-NEXT:     %3 = linalg.matmul ins(%arg0, %1 : tensor<100x10xi64>, tensor<10x8xi64>) outs(%2 : tensor<100x8xi64>) -> tensor<100x8xi64>
// CHECK-NEXT:     return %3 : tensor<100x8xi64>
// CHECK-NEXT:  }