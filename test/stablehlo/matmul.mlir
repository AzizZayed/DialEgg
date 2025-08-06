// RUN: %eggopt %s --eq-sat | FileCheck %s

func.func @_2mm(%x: tensor<100x10xi64>, %y: tensor<10x150xi64>, %z: tensor<150x8xi64>) -> tensor<100x8xi64> {
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

// CHECK: func.func @_2mm(%arg0: tensor<100x10xi64>, %arg1: tensor<10x150xi64>, %arg2: tensor<150x8xi64>) -> tensor<100x8xi64> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<10x150xi64>, tensor<150x8xi64>) -> tensor<10x8xi64>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<100x10xi64>, tensor<10x8xi64>) -> tensor<100x8xi64>
// CHECK-NEXT:     return %1 : tensor<100x8xi64>
// CHECK-NEXT: }

func.func @_3mm(%x: tensor<200x175xi64>, %y: tensor<175x250xi64>, %z: tensor<250x150xi64>, %w: tensor<150x10xi64>) -> tensor<200x10xi64> {
    %xy = stablehlo.dot_general %x, %y,
      batching_dims = [] x [],
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT],
      algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
      : (tensor<200x175xi64>, tensor<175x250xi64>) -> tensor<200x250xi64>
    
    %xy_z = stablehlo.dot_general %xy, %z,
      batching_dims = [] x [],
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT],
      algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
      : (tensor<200x250xi64>, tensor<250x150xi64>) -> tensor<200x150xi64>

    %xy_z__w = stablehlo.dot_general %xy_z, %w,
      batching_dims = [] x [],
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT],
      algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
      : (tensor<200x150xi64>, tensor<150x10xi64>) -> tensor<200x10xi64>

    func.return %xy_z__w : tensor<200x10xi64>
}

// CHECK: func.func @_3mm(%arg0: tensor<200x175xi64>, %arg1: tensor<175x250xi64>, %arg2: tensor<250x150xi64>, %arg3: tensor<150x10xi64>) -> tensor<200x10xi64> {
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<250x150xi64>, tensor<150x10xi64>) -> tensor<250x10xi64>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<175x250xi64>, tensor<250x10xi64>) -> tensor<175x10xi64>
// CHECK-NEXT:     %2 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<200x175xi64>, tensor<175x10xi64>) -> tensor<200x10xi64>
// CHECK-NEXT:     return %2 : tensor<200x10xi64>
// CHECK-NEXT: }