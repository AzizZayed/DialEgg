// RUN: %eggopt %s --eq-sat | FileCheck %s

func.func @matmul(%x: tensor<100x10xi64>, %y: tensor<10x150xi64>, %z: tensor<150x8xi64>) -> tensor<100x8xi64> {
    // (xy) z cost ac(b+d) = 100*150*(10+8) = 270,000
    // x (yz) cost bd(c+a) = 10*8*(150+100) = 20,000
    
    %xy = stablehlo.dot_general %x, %y,
      batching_dims = [] x [],
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT],
      algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
      : (tensor<100x10xi64>, tensor<10x150xi64>) -> tensor<100x150xi64>
    
    %xy_z = stablehlo.dot_general %xy, %z,
      batching_dims = [] x [],
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT],
      algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
      : (tensor<100x150xi64>, tensor<150x8xi64>) -> tensor<100x8xi64>

    func.return %xy_z : tensor<100x8xi64>
}

// CHECK: func.func @matmul(%arg0: tensor<100x10xi64>, %arg1: tensor<10x150xi64>, %arg2: tensor<150x8xi64>) -> tensor<100x8xi64> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<10x150xi64>, tensor<150x8xi64>) -> tensor<10x8xi64>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<100x10xi64>, tensor<10x8xi64>) -> tensor<100x8xi64>
// CHECK-NEXT:     return %1 : tensor<100x8xi64>
// CHECK-NEXT: }
