module @jit_gmm_objective attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<25xf32>, %arg1: tensor<25x10xf32>, %arg2: tensor<25x55xf32>, %arg3: tensor<1000x10xf32>, %arg4: tensor<f32>, %arg5: tensor<i32>) -> (tensor<f32> {jax.result_info = "result"}) {
    %0 = call @gmm_objective(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<25xf32>, tensor<25x10xf32>, tensor<25x55xf32>, tensor<1000x10xf32>, tensor<f32>, tensor<i32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @gmm_objective(%arg0: tensor<25xf32>, %arg1: tensor<25x10xf32>, %arg2: tensor<25x55xf32>, %arg3: tensor<1000x10xf32>, %arg4: tensor<f32>, %arg5: tensor<i32>) -> tensor<f32> {
    %0 = stablehlo.slice %arg2 [0:25, 0:10] : (tensor<25x55xf32>) -> tensor<25x10xf32>
    %1 = stablehlo.exponential %0 : tensor<25x10xf32>
    %2 = stablehlo.slice %arg2 [0:25, 0:10] : (tensor<25x55xf32>) -> tensor<25x10xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<25x10xf32>, tensor<f32>) -> tensor<25xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<10x10xf32>
    %5 = call @triu(%4) : (tensor<10x10xf32>) -> tensor<10x10xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<10x10xf32>
    %7 = stablehlo.compare  NE, %5, %6,  FLOAT : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xi1>
    %8 = call @cumsum(%7) : (tensor<10x10xi1>) -> tensor<100xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %10 = call @clip(%8, %c_2) : (tensor<100xi32>, tensor<i32>) -> tensor<100xi32>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<100xi32>
    %12 = stablehlo.compare  LT, %10, %11,  SIGNED : (tensor<100xi32>, tensor<100xi32>) -> tensor<100xi1>
    %c_4 = stablehlo.constant dense<45> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<100xi32>
    %14 = stablehlo.add %10, %13 : tensor<100xi32>
    %15 = stablehlo.select %12, %14, %10 : tensor<100xi1>, tensor<100xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<100xi32>) -> tensor<100x1xi32>
    %17 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<100xi32>
    %18 = "stablehlo.scatter"(%9, %16, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg6: tensor<i32>, %arg7: tensor<i32>):
      %140 = stablehlo.add %arg6, %arg7 : tensor<i32>
      stablehlo.return %140 : tensor<i32>
    }) : (tensor<45xi32>, tensor<100x1xi32>, tensor<100xi32>) -> tensor<45xi32>
    %19 = call @cumsum_1(%18) : (tensor<45xi32>) -> tensor<45xi32>
    %c_5 = stablehlo.constant dense<10> : tensor<i32>
    %20 = call @floor_divide(%19, %c_5) : (tensor<45xi32>, tensor<i32>) -> tensor<45xi32>
    %c_6 = stablehlo.constant dense<10> : tensor<i32>
    %21 = call @remainder(%20, %c_6) : (tensor<45xi32>, tensor<i32>) -> tensor<45xi32>
    %c_7 = stablehlo.constant dense<1> : tensor<i32>
    %22 = call @floor_divide(%19, %c_7) : (tensor<45xi32>, tensor<i32>) -> tensor<45xi32>
    %23 = call @remainder(%22, %c_6) : (tensor<45xi32>, tensor<i32>) -> tensor<45xi32>
    %24 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<25x10x10xf32>
    %25 = stablehlo.slice %arg2 [0:25, 10:55] : (tensor<25x55xf32>) -> tensor<25x45xf32>
    %26 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %27 = stablehlo.compare  LT, %21, %26,  SIGNED : (tensor<45xi32>, tensor<45xi32>) -> tensor<45xi1>
    %c_8 = stablehlo.constant dense<10> : tensor<i32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %29 = stablehlo.add %21, %28 : tensor<45xi32>
    %30 = stablehlo.select %27, %29, %21 : tensor<45xi1>, tensor<45xi32>
    %31 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %32 = stablehlo.compare  LT, %23, %31,  SIGNED : (tensor<45xi32>, tensor<45xi32>) -> tensor<45xi1>
    %33 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %34 = stablehlo.add %23, %33 : tensor<45xi32>
    %35 = stablehlo.select %32, %34, %23 : tensor<45xi1>, tensor<45xi32>
    %36 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<45xi32>) -> tensor<45x1xi32>
    %37 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<45xi32>) -> tensor<45x1xi32>
    %38 = stablehlo.concatenate %36, %37, dim = 1 : (tensor<45x1xi32>, tensor<45x1xi32>) -> tensor<45x2xi32>
    %39 = "stablehlo.scatter"(%24, %38, %25) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
      stablehlo.return %arg7 : tensor<f32>
    }) : (tensor<25x10x10xf32>, tensor<45x2xi32>, tensor<25x45xf32>) -> tensor<25x10x10xf32>
    %40 = stablehlo.transpose %39, dims = [0, 2, 1] : (tensor<25x10x10xf32>) -> tensor<25x10x10xf32>
    %41 = stablehlo.broadcast_in_dim %arg3, dims = [0, 2] : (tensor<1000x10xf32>) -> tensor<1000x1x10xf32>
    %42 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<25x10xf32>) -> tensor<1x25x10xf32>
    %43 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2] : (tensor<1000x1x10xf32>) -> tensor<1000x25x10xf32>
    %44 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x25x10xf32>) -> tensor<1000x25x10xf32>
    %45 = stablehlo.subtract %43, %44 : tensor<1000x25x10xf32>
    %46 = stablehlo.dot_general %45, %40, batching_dims = [1] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1000x25x10xf32>, tensor<25x10x10xf32>) -> tensor<25x1000x10xf32>
    %47 = stablehlo.transpose %46, dims = [1, 0, 2] : (tensor<25x1000x10xf32>) -> tensor<1000x25x10xf32>
    %48 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<25x10xf32>) -> tensor<1x25x10xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<1x25x10xf32>) -> tensor<1000x25x10xf32>
    %50 = stablehlo.multiply %49, %45 : tensor<1000x25x10xf32>
    %51 = stablehlo.add %50, %47 : tensor<1000x25x10xf32>
    %52 = stablehlo.multiply %51, %51 : tensor<1000x25x10xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %53 = stablehlo.reduce(%52 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<1000x25x10xf32>, tensor<f32>) -> tensor<1000x25xf32>
    %54 = stablehlo.add %arg0, %3 : tensor<25xf32>
    %cst_10 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1000x25xf32>
    %56 = stablehlo.multiply %55, %53 : tensor<1000x25xf32>
    %57 = stablehlo.broadcast_in_dim %54, dims = [1] : (tensor<25xf32>) -> tensor<1x25xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1] : (tensor<1x25xf32>) -> tensor<1000x25xf32>
    %59 = stablehlo.subtract %58, %56 : tensor<1000x25xf32>
    %cst_11 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %60 = stablehlo.reduce(%59 init: %cst_11) applies stablehlo.maximum across dimensions = [1] : (tensor<1000x25xf32>, tensor<f32>) -> tensor<1000xf32>
    %cst_12 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %62 = stablehlo.maximum %61, %60 : tensor<1000xf32>
    %63 = stablehlo.is_finite %62 : (tensor<1000xf32>) -> tensor<1000xi1>
    %64 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %65 = stablehlo.select %63, %62, %64 : tensor<1000xi1>, tensor<1000xf32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [0] : (tensor<1000xf32>) -> tensor<1000x1xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<1000x1xf32>) -> tensor<1000x25xf32>
    %68 = stablehlo.subtract %59, %67 : tensor<1000x25xf32>
    %69 = stablehlo.exponential %68 : tensor<1000x25xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %70 = stablehlo.reduce(%69 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<1000x25xf32>, tensor<f32>) -> tensor<1000xf32>
    %71 = stablehlo.abs %70 : tensor<1000xf32>
    %72 = stablehlo.log %71 : tensor<1000xf32>
    %73 = stablehlo.add %72, %65 : tensor<1000xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %74 = stablehlo.reduce(%73 init: %cst_14) applies stablehlo.add across dimensions = [0] : (tensor<1000xf32>, tensor<f32>) -> tensor<f32>
    %cst_15 = stablehlo.constant dense<6.28318548> : tensor<f32>
    %75 = stablehlo.log %cst_15 : tensor<f32>
    %cst_16 = stablehlo.constant dense<-5.000000e+03> : tensor<f32>
    %76 = stablehlo.multiply %cst_16, %75 : tensor<f32>
    %77 = stablehlo.convert %76 : tensor<f32>
    %78 = stablehlo.add %77, %74 : tensor<f32>
    %cst_17 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %79 = stablehlo.reduce(%arg0 init: %cst_17) applies stablehlo.maximum across dimensions = [0] : (tensor<25xf32>, tensor<f32>) -> tensor<f32>
    %80 = stablehlo.maximum %cst_12, %79 : tensor<f32>
    %81 = stablehlo.is_finite %80 : (tensor<f32>) -> tensor<i1>
    %82 = stablehlo.select %81, %80, %cst_1 : tensor<i1>, tensor<f32>
    %83 = stablehlo.broadcast_in_dim %82, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0] : (tensor<1xf32>) -> tensor<25xf32>
    %85 = stablehlo.subtract %arg0, %84 : tensor<25xf32>
    %86 = stablehlo.exponential %85 : tensor<25xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %87 = stablehlo.reduce(%86 init: %cst_18) applies stablehlo.add across dimensions = [0] : (tensor<25xf32>, tensor<f32>) -> tensor<f32>
    %88 = stablehlo.abs %87 : tensor<f32>
    %89 = stablehlo.log %88 : tensor<f32>
    %90 = stablehlo.add %89, %82 : tensor<f32>
    %cst_19 = stablehlo.constant dense<1.000000e+03> : tensor<f32>
    %91 = stablehlo.multiply %cst_19, %90 : tensor<f32>
    %92 = stablehlo.subtract %78, %91 : tensor<f32>
    %93 = stablehlo.add %c_6, %arg5 : tensor<i32>
    %c_20 = stablehlo.constant dense<1> : tensor<i32>
    %94 = stablehlo.add %93, %c_20 : tensor<i32>
    %cst_21 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %95 = stablehlo.multiply %cst_21, %arg4 : tensor<f32>
    %96 = stablehlo.multiply %95, %arg4 : tensor<f32>
    %97 = stablehlo.multiply %1, %1 : tensor<25x10xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %98 = stablehlo.reduce(%97 init: %cst_22) applies stablehlo.add across dimensions = [1] : (tensor<25x10xf32>, tensor<f32>) -> tensor<25xf32>
    %99 = stablehlo.slice %arg2 [0:25, 10:55] : (tensor<25x55xf32>) -> tensor<25x45xf32>
    %100 = stablehlo.multiply %99, %99 : tensor<25x45xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %101 = stablehlo.reduce(%100 init: %cst_23) applies stablehlo.add across dimensions = [1] : (tensor<25x45xf32>, tensor<f32>) -> tensor<25xf32>
    %102 = stablehlo.add %98, %101 : tensor<25xf32>
    %103 = stablehlo.convert %96 : tensor<f32>
    %104 = stablehlo.broadcast_in_dim %103, dims = [] : (tensor<f32>) -> tensor<25xf32>
    %105 = stablehlo.multiply %104, %102 : tensor<25xf32>
    %106 = stablehlo.convert %arg5 : (tensor<i32>) -> tensor<f32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<25xf32>
    %108 = stablehlo.multiply %107, %3 : tensor<25xf32>
    %109 = stablehlo.subtract %105, %108 : tensor<25xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %110 = stablehlo.reduce(%109 init: %cst_24) applies stablehlo.add across dimensions = [0] : (tensor<25xf32>, tensor<f32>) -> tensor<f32>
    %111 = stablehlo.multiply %94, %c_6 : tensor<i32>
    %cst_25 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %112 = stablehlo.sqrt %cst_25 : tensor<f32>
    %113 = stablehlo.divide %arg4, %112 : tensor<f32>
    %114 = stablehlo.log %113 : tensor<f32>
    %115 = stablehlo.convert %111 : (tensor<i32>) -> tensor<f32>
    %116 = stablehlo.multiply %115, %114 : tensor<f32>
    %117 = stablehlo.convert %94 : (tensor<i32>) -> tensor<f32>
    %118 = stablehlo.multiply %cst_21, %117 : tensor<f32>
    %cst_26 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %cst_27 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %119 = stablehlo.multiply %cst_26, %cst_27 : tensor<f32>
    %cst_28 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %120 = stablehlo.subtract %cst_27, %cst_28 : tensor<f32>
    %121 = stablehlo.multiply %119, %120 : tensor<f32>
    %cst_29 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %122 = stablehlo.log %cst_29 : tensor<f32>
    %123 = stablehlo.multiply %121, %122 : tensor<f32>
    %124 = stablehlo.iota dim = 0 : tensor<10xf32>
    %125 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %126 = stablehlo.divide %124, %125 : tensor<10xf32>
    %127 = stablehlo.broadcast_in_dim %118, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %128 = stablehlo.convert %127 : tensor<1xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0] : (tensor<1xf32>) -> tensor<10xf32>
    %130 = stablehlo.subtract %129, %126 : tensor<10xf32>
    %131 = chlo.lgamma %130 : tensor<10xf32> -> tensor<10xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %132 = stablehlo.reduce(%131 init: %cst_30) applies stablehlo.add across dimensions = [0] : (tensor<10xf32>, tensor<f32>) -> tensor<f32>
    %133 = stablehlo.convert %123 : tensor<f32>
    %134 = stablehlo.add %132, %133 : tensor<f32>
    %135 = stablehlo.convert %116 : tensor<f32>
    %136 = stablehlo.subtract %135, %134 : tensor<f32>
    %cst_31 = stablehlo.constant dense<2.500000e+01> : tensor<f32>
    %137 = stablehlo.multiply %cst_31, %136 : tensor<f32>
    %138 = stablehlo.subtract %110, %137 : tensor<f32>
    %139 = stablehlo.add %92, %138 : tensor<f32>
    return %139 : tensor<f32>
  }
  func.func private @triu(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<10x10xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<10x10xi32>
    %2 = stablehlo.add %0, %1 : tensor<10x10xi32>
    %3 = stablehlo.iota dim = 1 : tensor<10x10xi32>
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<10x10xf32>
    %6 = stablehlo.select %4, %5, %arg0 : tensor<10x10xi1>, tensor<10x10xf32>
    return %6 : tensor<10x10xf32>
  }
  func.func private @cumsum(%arg0: tensor<10x10xi1>) -> tensor<100xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<10x10xi1>) -> tensor<100xi1>
    %1 = stablehlo.convert %0 : (tensor<100xi1>) -> tensor<100xi32>
    %2 = call @cumsum_0(%1) : (tensor<100xi32>) -> tensor<100xi32>
    return %2 : tensor<100xi32>
  }
  func.func private @cumsum_0(%arg0: tensor<100xi32>) -> tensor<100xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{base_dilations = array<i64: 1>, padding = dense<[[99, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 100>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) : (tensor<100xi32>, tensor<i32>) -> tensor<100xi32>
    return %1 : tensor<100xi32>
  }
  func.func private @clip(%arg0: tensor<100xi32>, %arg1: tensor<i32>) -> tensor<100xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<100xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<100xi32>
    return %2 : tensor<100xi32>
  }
  func.func private @cumsum_1(%arg0: tensor<45xi32>) -> tensor<45xi32> {
    %0 = call @cumsum_2(%arg0) : (tensor<45xi32>) -> tensor<45xi32>
    return %0 : tensor<45xi32>
  }
  func.func private @cumsum_2(%arg0: tensor<45xi32>) -> tensor<45xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{base_dilations = array<i64: 1>, padding = dense<[[44, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 45>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) : (tensor<45xi32>, tensor<i32>) -> tensor<45xi32>
    return %1 : tensor<45xi32>
  }
  func.func private @floor_divide(%arg0: tensor<45xi32>, %arg1: tensor<i32>) -> tensor<45xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<45xi32>
    %2 = stablehlo.sign %arg0 : tensor<45xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<45xi32>, tensor<45xi32>) -> tensor<45xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<45xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<45xi32>, tensor<45xi32>) -> tensor<45xi1>
    %10 = stablehlo.and %5, %9 : tensor<45xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<45xi32>
    %13 = call @_where(%10, %12, %1) : (tensor<45xi1>, tensor<45xi32>, tensor<45xi32>) -> tensor<45xi32>
    return %13 : tensor<45xi32>
  }
  func.func private @_where(%arg0: tensor<45xi1>, %arg1: tensor<45xi32>, %arg2: tensor<45xi32>) -> tensor<45xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<45xi1>, tensor<45xi32>
    return %0 : tensor<45xi32>
  }
  func.func private @remainder(%arg0: tensor<45xi32>, %arg1: tensor<i32>) -> tensor<45xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  EQ, %0, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %2 = call @_where_3(%1, %c_0, %0) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %4 = stablehlo.remainder %arg0, %3 : tensor<45xi32>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %6 = stablehlo.compare  NE, %4, %5,  SIGNED : (tensor<45xi32>, tensor<45xi32>) -> tensor<45xi1>
    %7 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %8 = stablehlo.compare  LT, %4, %7,  SIGNED : (tensor<45xi32>, tensor<45xi32>) -> tensor<45xi1>
    %9 = stablehlo.compare  LT, %2, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<45xi1>
    %11 = stablehlo.compare  NE, %8, %10,  UNSIGNED : (tensor<45xi1>, tensor<45xi1>) -> tensor<45xi1>
    %12 = stablehlo.and %11, %6 : tensor<45xi1>
    %13 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<45xi32>
    %14 = stablehlo.add %4, %13 : tensor<45xi32>
    %15 = stablehlo.select %12, %14, %4 : tensor<45xi1>, tensor<45xi32>
    return %15 : tensor<45xi32>
  }
  func.func private @_where_3(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
}
