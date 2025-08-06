module @jit_kernel_heat3d attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<500x500x500xf32>) -> (tensor<500x500x500xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<[[0.000000e+00, 1.250000e-01, 0.000000e+00], [1.250000e-01, 2.500000e-01, 1.250000e-01], [0.000000e+00, 1.250000e-01, 0.000000e+00]]> : tensor<3x3xf32>
    %0 = stablehlo.iota dim = 0 : tensor<500x3xi32>
    %1 = stablehlo.iota dim = 0 : tensor<3x500xi32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<3x500xi32>) -> tensor<500x3xi32>
    %3 = stablehlo.add %0, %2 : tensor<500x3xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<500x3xi32>
    %5 = stablehlo.add %3, %4 : tensor<500x3xi32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<500x3xi32>
    %7 = stablehlo.subtract %5, %6 : tensor<500x3xi32>
    %8 = stablehlo.iota dim = 0 : tensor<500x3xi32>
    %9 = stablehlo.iota dim = 0 : tensor<3x500xi32>
    %10 = stablehlo.transpose %9, dims = [1, 0] : (tensor<3x500xi32>) -> tensor<500x3xi32>
    %11 = stablehlo.add %8, %10 : tensor<500x3xi32>
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<500x3xi32>
    %13 = stablehlo.add %11, %12 : tensor<500x3xi32>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<500x3xi32>
    %15 = stablehlo.subtract %13, %14 : tensor<500x3xi32>
    %16 = stablehlo.iota dim = 0 : tensor<500x3xi32>
    %17 = stablehlo.iota dim = 0 : tensor<3x500xi32>
    %18 = stablehlo.transpose %17, dims = [1, 0] : (tensor<3x500xi32>) -> tensor<500x3xi32>
    %19 = stablehlo.add %16, %18 : tensor<500x3xi32>
    %20 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<500x3xi32>
    %21 = stablehlo.add %19, %20 : tensor<500x3xi32>
    %22 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<500x3xi32>
    %23 = stablehlo.subtract %21, %22 : tensor<500x3xi32>
    %24 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<500x3xi32>) -> tensor<500x500x3xi32>
    %25 = stablehlo.broadcast_in_dim %15, dims = [0, 2] : (tensor<500x3xi32>) -> tensor<500x500x3xi32>
    %26 = stablehlo.broadcast_in_dim %24, dims = [0, 2, 3] : (tensor<500x500x3xi32>) -> tensor<500x500x500x3xi32>
    %27 = stablehlo.broadcast_in_dim %23, dims = [1, 2] : (tensor<500x3xi32>) -> tensor<500x500x3xi32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [1, 2, 3] : (tensor<500x500x3xi32>) -> tensor<500x500x500x3xi32>
    %29 = stablehlo.broadcast_in_dim %27, dims = [1, 2, 3] : (tensor<500x500x3xi32>) -> tensor<500x500x500x3xi32>
    %30 = stablehlo.reshape %26 : (tensor<500x500x500x3xi32>) -> tensor<125000000x3xi32>
    %31 = stablehlo.reshape %28 : (tensor<500x500x500x3xi32>) -> tensor<125000000x3xi32>
    %32 = stablehlo.reshape %29 : (tensor<500x500x500x3xi32>) -> tensor<125000000x3xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %33 = call @_pad(%arg0, %c_1) : (tensor<500x500x500xf32>, tensor<i32>) -> tensor<502x502x502xf32>
    %34 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<125000000x3xi32>) -> tensor<125000000x3x1x1xi32>
    %35 = stablehlo.broadcast_in_dim %31, dims = [0, 2] : (tensor<125000000x3xi32>) -> tensor<125000000x1x3x1xi32>
    %36 = stablehlo.broadcast_in_dim %32, dims = [0, 3] : (tensor<125000000x3xi32>) -> tensor<125000000x1x1x3xi32>
    %37 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<125000000x3x1x1xi32>
    %38 = stablehlo.compare  LT, %34, %37,  SIGNED : (tensor<125000000x3x1x1xi32>, tensor<125000000x3x1x1xi32>) -> tensor<125000000x3x1x1xi1>
    %c_2 = stablehlo.constant dense<502> : tensor<i32>
    %39 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<125000000x3x1x1xi32>
    %40 = stablehlo.add %34, %39 : tensor<125000000x3x1x1xi32>
    %41 = stablehlo.select %38, %40, %34 : tensor<125000000x3x1x1xi1>, tensor<125000000x3x1x1xi32>
    %42 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<125000000x1x3x1xi32>
    %43 = stablehlo.compare  LT, %35, %42,  SIGNED : (tensor<125000000x1x3x1xi32>, tensor<125000000x1x3x1xi32>) -> tensor<125000000x1x3x1xi1>
    %44 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<125000000x1x3x1xi32>
    %45 = stablehlo.add %35, %44 : tensor<125000000x1x3x1xi32>
    %46 = stablehlo.select %43, %45, %35 : tensor<125000000x1x3x1xi1>, tensor<125000000x1x3x1xi32>
    %47 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<125000000x1x1x3xi32>
    %48 = stablehlo.compare  LT, %36, %47,  SIGNED : (tensor<125000000x1x1x3xi32>, tensor<125000000x1x1x3xi32>) -> tensor<125000000x1x1x3xi1>
    %49 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<125000000x1x1x3xi32>
    %50 = stablehlo.add %36, %49 : tensor<125000000x1x1x3xi32>
    %51 = stablehlo.select %48, %50, %36 : tensor<125000000x1x1x3xi1>, tensor<125000000x1x1x3xi32>
    %52 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<125000000x3x1x1xi32>) -> tensor<125000000x3x3x3xi32>
    %53 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2, 3] : (tensor<125000000x1x3x1xi32>) -> tensor<125000000x3x3x3xi32>
    %54 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2, 3] : (tensor<125000000x1x1x3xi32>) -> tensor<125000000x3x3x3xi32>
    %55 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<125000000x3x3x3xi32>) -> tensor<125000000x3x3x3x1xi32>
    %56 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2, 3] : (tensor<125000000x3x3x3xi32>) -> tensor<125000000x3x3x3x1xi32>
    %57 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<125000000x3x3x3xi32>) -> tensor<125000000x3x3x3x1xi32>
    %58 = stablehlo.concatenate %55, %56, %57, dim = 4 : (tensor<125000000x3x3x3x1xi32>, tensor<125000000x3x3x3x1xi32>, tensor<125000000x3x3x3x1xi32>) -> tensor<125000000x3x3x3x3xi32>
    %59 = "stablehlo.gather"(%33, %58) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 4>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<502x502x502xf32>, tensor<125000000x3x3x3x3xi32>) -> tensor<125000000x3x3x3xf32>
    %60 = stablehlo.broadcast_in_dim %cst, dims = [1, 2] : (tensor<3x3xf32>) -> tensor<1x3x3xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [1, 2, 3] : (tensor<1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<1x1x3x3xf32>) -> tensor<125000000x3x3x3xf32>
    %63 = stablehlo.multiply %59, %62 : tensor<125000000x3x3x3xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %64 = stablehlo.reduce(%63 init: %cst_3) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<125000000x3x3x3xf32>, tensor<f32>) -> tensor<125000000xf32>
    %65 = stablehlo.reshape %64 : (tensor<125000000xf32>) -> tensor<500x500x500xf32>
    return %65 : tensor<500x500x500xf32>
  }
  func.func private @_pad(%arg0: tensor<500x500x500xf32>, %arg1: tensor<i32>) -> tensor<502x502x502xf32> {
    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.pad %arg0, %0, low = [0, 0, 0], high = [2, 2, 2], interior = [0, 0, 0] : (tensor<500x500x500xf32>, tensor<f32>) -> tensor<502x502x502xf32>
    return %1 : tensor<502x502x502xf32>
  }
}
