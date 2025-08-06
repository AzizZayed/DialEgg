module @jit_sfn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func private @relu(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<512xf32>
    return %1 : tensor<512xf32>
  }
  func.func public @main(%arg0: tensor<5x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>, %arg4: tensor<512x512xf32>, %arg5: tensor<512x512xf32>, %arg6: tensor<512x512xf32>, %arg7: tensor<512x512xf32>, %arg8: tensor<512x512xf32>, %arg9: tensor<512x512xf32>, %arg10: tensor<512x512xf32>, %arg11: tensor<512x512xf32>, %arg12: tensor<512x512xf32>, %arg13: tensor<512x512xf32>, %arg14: tensor<512x512xf32>, %arg15: tensor<512x512xf32>, %arg16: tensor<512x512xf32>, %arg17: tensor<512xf32>) -> (tensor<512xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.slice %arg0 [0:1, 0:512] : (tensor<5x512xf32>) -> tensor<1x512xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %3 = stablehlo.dot_general %arg17, %arg9, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.dot_general %1, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %6 = stablehlo.dot_general %arg17, %arg10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %7 = stablehlo.add %5, %6 : tensor<512xf32>
    %8 = stablehlo.dot_general %1, %arg3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %9 = stablehlo.dot_general %arg17, %arg11, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %10 = stablehlo.add %8, %9 : tensor<512xf32>
    %11 = stablehlo.dot_general %1, %arg4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %12 = stablehlo.dot_general %arg17, %arg12, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %13 = stablehlo.add %11, %12 : tensor<512xf32>
    %14 = stablehlo.dot_general %1, %arg5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %15 = stablehlo.dot_general %arg17, %arg13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %16 = stablehlo.add %14, %15 : tensor<512xf32>
    %17 = stablehlo.dot_general %1, %arg6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %18 = stablehlo.dot_general %arg17, %arg14, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %19 = stablehlo.add %17, %18 : tensor<512xf32>
    %20 = stablehlo.dot_general %1, %arg7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %21 = stablehlo.dot_general %arg17, %arg15, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %22 = stablehlo.add %20, %21 : tensor<512xf32>
    %23 = stablehlo.dot_general %1, %arg8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %24 = stablehlo.dot_general %arg17, %arg16, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %25 = stablehlo.add %23, %24 : tensor<512xf32>
    %26 = call @relu(%4) : (tensor<512xf32>) -> tensor<512xf32>
    %27 = stablehlo.negate %13 : tensor<512xf32>
    %28 = stablehlo.exponential %27 : tensor<512xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %30 = stablehlo.add %29, %28 : tensor<512xf32>
    %31 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %32 = stablehlo.divide %31, %30 : tensor<512xf32>
    %33 = stablehlo.add %26, %32 : tensor<512xf32>
    %34 = stablehlo.negate %7 : tensor<512xf32>
    %35 = stablehlo.exponential %34 : tensor<512xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %37 = stablehlo.add %36, %35 : tensor<512xf32>
    %38 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %39 = stablehlo.divide %38, %37 : tensor<512xf32>
    %40 = stablehlo.tanh %10 : tensor<512xf32>
    %41 = stablehlo.add %39, %40 : tensor<512xf32>
    %42 = stablehlo.negate %16 : tensor<512xf32>
    %43 = stablehlo.exponential %42 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %45 = stablehlo.add %44, %43 : tensor<512xf32>
    %46 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %47 = stablehlo.divide %46, %45 : tensor<512xf32>
    %48 = stablehlo.tanh %19 : tensor<512xf32>
    %49 = stablehlo.multiply %47, %48 : tensor<512xf32>
    %50 = stablehlo.negate %22 : tensor<512xf32>
    %51 = stablehlo.exponential %50 : tensor<512xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %53 = stablehlo.add %52, %51 : tensor<512xf32>
    %54 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %55 = stablehlo.divide %54, %53 : tensor<512xf32>
    %56 = call @relu(%25) : (tensor<512xf32>) -> tensor<512xf32>
    %57 = stablehlo.multiply %55, %56 : tensor<512xf32>
    %58 = stablehlo.negate %41 : tensor<512xf32>
    %59 = stablehlo.exponential %58 : tensor<512xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %61 = stablehlo.add %60, %59 : tensor<512xf32>
    %62 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %63 = stablehlo.divide %62, %61 : tensor<512xf32>
    %64 = stablehlo.tanh %49 : tensor<512xf32>
    %65 = stablehlo.add %63, %64 : tensor<512xf32>
    %66 = stablehlo.tanh %33 : tensor<512xf32>
    %67 = stablehlo.tanh %57 : tensor<512xf32>
    %68 = stablehlo.multiply %66, %67 : tensor<512xf32>
    %69 = stablehlo.tanh %65 : tensor<512xf32>
    %70 = stablehlo.tanh %68 : tensor<512xf32>
    %71 = stablehlo.multiply %69, %70 : tensor<512xf32>
    %72 = stablehlo.tanh %71 : tensor<512xf32>
    %73 = stablehlo.slice %arg0 [1:2, 0:512] : (tensor<5x512xf32>) -> tensor<1x512xf32>
    %74 = stablehlo.reshape %73 : (tensor<1x512xf32>) -> tensor<512xf32>
    %75 = stablehlo.dot_general %74, %arg1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %76 = stablehlo.dot_general %72, %arg9, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %77 = stablehlo.add %75, %76 : tensor<512xf32>
    %78 = stablehlo.dot_general %74, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %79 = stablehlo.dot_general %72, %arg10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %80 = stablehlo.add %78, %79 : tensor<512xf32>
    %81 = stablehlo.dot_general %74, %arg3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %82 = stablehlo.dot_general %72, %arg11, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %83 = stablehlo.add %81, %82 : tensor<512xf32>
    %84 = stablehlo.dot_general %74, %arg4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %85 = stablehlo.dot_general %72, %arg12, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %86 = stablehlo.add %84, %85 : tensor<512xf32>
    %87 = stablehlo.dot_general %74, %arg5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %88 = stablehlo.dot_general %72, %arg13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %89 = stablehlo.add %87, %88 : tensor<512xf32>
    %90 = stablehlo.dot_general %74, %arg6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %91 = stablehlo.dot_general %72, %arg14, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %92 = stablehlo.add %90, %91 : tensor<512xf32>
    %93 = stablehlo.dot_general %74, %arg7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %94 = stablehlo.dot_general %72, %arg15, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %95 = stablehlo.add %93, %94 : tensor<512xf32>
    %96 = stablehlo.dot_general %74, %arg8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %97 = stablehlo.dot_general %72, %arg16, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %98 = stablehlo.add %96, %97 : tensor<512xf32>
    %99 = call @relu(%77) : (tensor<512xf32>) -> tensor<512xf32>
    %100 = stablehlo.negate %86 : tensor<512xf32>
    %101 = stablehlo.exponential %100 : tensor<512xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %102 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %103 = stablehlo.add %102, %101 : tensor<512xf32>
    %104 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %105 = stablehlo.divide %104, %103 : tensor<512xf32>
    %106 = stablehlo.add %99, %105 : tensor<512xf32>
    %107 = stablehlo.negate %80 : tensor<512xf32>
    %108 = stablehlo.exponential %107 : tensor<512xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %109 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %110 = stablehlo.add %109, %108 : tensor<512xf32>
    %111 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %112 = stablehlo.divide %111, %110 : tensor<512xf32>
    %113 = stablehlo.tanh %83 : tensor<512xf32>
    %114 = stablehlo.add %112, %113 : tensor<512xf32>
    %115 = stablehlo.negate %89 : tensor<512xf32>
    %116 = stablehlo.exponential %115 : tensor<512xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %118 = stablehlo.add %117, %116 : tensor<512xf32>
    %119 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %120 = stablehlo.divide %119, %118 : tensor<512xf32>
    %121 = stablehlo.tanh %92 : tensor<512xf32>
    %122 = stablehlo.multiply %120, %121 : tensor<512xf32>
    %123 = stablehlo.negate %95 : tensor<512xf32>
    %124 = stablehlo.exponential %123 : tensor<512xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %125 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %126 = stablehlo.add %125, %124 : tensor<512xf32>
    %127 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %128 = stablehlo.divide %127, %126 : tensor<512xf32>
    %129 = call @relu(%98) : (tensor<512xf32>) -> tensor<512xf32>
    %130 = stablehlo.multiply %128, %129 : tensor<512xf32>
    %131 = stablehlo.negate %114 : tensor<512xf32>
    %132 = stablehlo.exponential %131 : tensor<512xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %133 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %134 = stablehlo.add %133, %132 : tensor<512xf32>
    %135 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %136 = stablehlo.divide %135, %134 : tensor<512xf32>
    %137 = stablehlo.tanh %122 : tensor<512xf32>
    %138 = stablehlo.add %136, %137 : tensor<512xf32>
    %139 = stablehlo.tanh %106 : tensor<512xf32>
    %140 = stablehlo.tanh %130 : tensor<512xf32>
    %141 = stablehlo.multiply %139, %140 : tensor<512xf32>
    %142 = stablehlo.tanh %138 : tensor<512xf32>
    %143 = stablehlo.tanh %141 : tensor<512xf32>
    %144 = stablehlo.multiply %142, %143 : tensor<512xf32>
    %145 = stablehlo.tanh %144 : tensor<512xf32>
    %146 = stablehlo.slice %arg0 [2:3, 0:512] : (tensor<5x512xf32>) -> tensor<1x512xf32>
    %147 = stablehlo.reshape %146 : (tensor<1x512xf32>) -> tensor<512xf32>
    %148 = stablehlo.dot_general %147, %arg1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %149 = stablehlo.dot_general %145, %arg9, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %150 = stablehlo.add %148, %149 : tensor<512xf32>
    %151 = stablehlo.dot_general %147, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %152 = stablehlo.dot_general %145, %arg10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %153 = stablehlo.add %151, %152 : tensor<512xf32>
    %154 = stablehlo.dot_general %147, %arg3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %155 = stablehlo.dot_general %145, %arg11, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %156 = stablehlo.add %154, %155 : tensor<512xf32>
    %157 = stablehlo.dot_general %147, %arg4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %158 = stablehlo.dot_general %145, %arg12, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %159 = stablehlo.add %157, %158 : tensor<512xf32>
    %160 = stablehlo.dot_general %147, %arg5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %161 = stablehlo.dot_general %145, %arg13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %162 = stablehlo.add %160, %161 : tensor<512xf32>
    %163 = stablehlo.dot_general %147, %arg6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %164 = stablehlo.dot_general %145, %arg14, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %165 = stablehlo.add %163, %164 : tensor<512xf32>
    %166 = stablehlo.dot_general %147, %arg7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %167 = stablehlo.dot_general %145, %arg15, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %168 = stablehlo.add %166, %167 : tensor<512xf32>
    %169 = stablehlo.dot_general %147, %arg8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %170 = stablehlo.dot_general %145, %arg16, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %171 = stablehlo.add %169, %170 : tensor<512xf32>
    %172 = call @relu(%150) : (tensor<512xf32>) -> tensor<512xf32>
    %173 = stablehlo.negate %159 : tensor<512xf32>
    %174 = stablehlo.exponential %173 : tensor<512xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %175 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %176 = stablehlo.add %175, %174 : tensor<512xf32>
    %177 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %178 = stablehlo.divide %177, %176 : tensor<512xf32>
    %179 = stablehlo.add %172, %178 : tensor<512xf32>
    %180 = stablehlo.negate %153 : tensor<512xf32>
    %181 = stablehlo.exponential %180 : tensor<512xf32>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %182 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %183 = stablehlo.add %182, %181 : tensor<512xf32>
    %184 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %185 = stablehlo.divide %184, %183 : tensor<512xf32>
    %186 = stablehlo.tanh %156 : tensor<512xf32>
    %187 = stablehlo.add %185, %186 : tensor<512xf32>
    %188 = stablehlo.negate %162 : tensor<512xf32>
    %189 = stablehlo.exponential %188 : tensor<512xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %190 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %191 = stablehlo.add %190, %189 : tensor<512xf32>
    %192 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %193 = stablehlo.divide %192, %191 : tensor<512xf32>
    %194 = stablehlo.tanh %165 : tensor<512xf32>
    %195 = stablehlo.multiply %193, %194 : tensor<512xf32>
    %196 = stablehlo.negate %168 : tensor<512xf32>
    %197 = stablehlo.exponential %196 : tensor<512xf32>
    %cst_12 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %199 = stablehlo.add %198, %197 : tensor<512xf32>
    %200 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %201 = stablehlo.divide %200, %199 : tensor<512xf32>
    %202 = call @relu(%171) : (tensor<512xf32>) -> tensor<512xf32>
    %203 = stablehlo.multiply %201, %202 : tensor<512xf32>
    %204 = stablehlo.negate %187 : tensor<512xf32>
    %205 = stablehlo.exponential %204 : tensor<512xf32>
    %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %206 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %207 = stablehlo.add %206, %205 : tensor<512xf32>
    %208 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %209 = stablehlo.divide %208, %207 : tensor<512xf32>
    %210 = stablehlo.tanh %195 : tensor<512xf32>
    %211 = stablehlo.add %209, %210 : tensor<512xf32>
    %212 = stablehlo.tanh %179 : tensor<512xf32>
    %213 = stablehlo.tanh %203 : tensor<512xf32>
    %214 = stablehlo.multiply %212, %213 : tensor<512xf32>
    %215 = stablehlo.tanh %211 : tensor<512xf32>
    %216 = stablehlo.tanh %214 : tensor<512xf32>
    %217 = stablehlo.multiply %215, %216 : tensor<512xf32>
    %218 = stablehlo.tanh %217 : tensor<512xf32>
    %219 = stablehlo.slice %arg0 [3:4, 0:512] : (tensor<5x512xf32>) -> tensor<1x512xf32>
    %220 = stablehlo.reshape %219 : (tensor<1x512xf32>) -> tensor<512xf32>
    %221 = stablehlo.dot_general %220, %arg1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %222 = stablehlo.dot_general %218, %arg9, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %223 = stablehlo.add %221, %222 : tensor<512xf32>
    %224 = stablehlo.dot_general %220, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %225 = stablehlo.dot_general %218, %arg10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %226 = stablehlo.add %224, %225 : tensor<512xf32>
    %227 = stablehlo.dot_general %220, %arg3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %228 = stablehlo.dot_general %218, %arg11, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %229 = stablehlo.add %227, %228 : tensor<512xf32>
    %230 = stablehlo.dot_general %220, %arg4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %231 = stablehlo.dot_general %218, %arg12, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %232 = stablehlo.add %230, %231 : tensor<512xf32>
    %233 = stablehlo.dot_general %220, %arg5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %234 = stablehlo.dot_general %218, %arg13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %235 = stablehlo.add %233, %234 : tensor<512xf32>
    %236 = stablehlo.dot_general %220, %arg6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %237 = stablehlo.dot_general %218, %arg14, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %238 = stablehlo.add %236, %237 : tensor<512xf32>
    %239 = stablehlo.dot_general %220, %arg7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %240 = stablehlo.dot_general %218, %arg15, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %241 = stablehlo.add %239, %240 : tensor<512xf32>
    %242 = stablehlo.dot_general %220, %arg8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %243 = stablehlo.dot_general %218, %arg16, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %244 = stablehlo.add %242, %243 : tensor<512xf32>
    %245 = call @relu(%223) : (tensor<512xf32>) -> tensor<512xf32>
    %246 = stablehlo.negate %232 : tensor<512xf32>
    %247 = stablehlo.exponential %246 : tensor<512xf32>
    %cst_14 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %248 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %249 = stablehlo.add %248, %247 : tensor<512xf32>
    %250 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %251 = stablehlo.divide %250, %249 : tensor<512xf32>
    %252 = stablehlo.add %245, %251 : tensor<512xf32>
    %253 = stablehlo.negate %226 : tensor<512xf32>
    %254 = stablehlo.exponential %253 : tensor<512xf32>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %255 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %256 = stablehlo.add %255, %254 : tensor<512xf32>
    %257 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %258 = stablehlo.divide %257, %256 : tensor<512xf32>
    %259 = stablehlo.tanh %229 : tensor<512xf32>
    %260 = stablehlo.add %258, %259 : tensor<512xf32>
    %261 = stablehlo.negate %235 : tensor<512xf32>
    %262 = stablehlo.exponential %261 : tensor<512xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %263 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %264 = stablehlo.add %263, %262 : tensor<512xf32>
    %265 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %266 = stablehlo.divide %265, %264 : tensor<512xf32>
    %267 = stablehlo.tanh %238 : tensor<512xf32>
    %268 = stablehlo.multiply %266, %267 : tensor<512xf32>
    %269 = stablehlo.negate %241 : tensor<512xf32>
    %270 = stablehlo.exponential %269 : tensor<512xf32>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %271 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %272 = stablehlo.add %271, %270 : tensor<512xf32>
    %273 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %274 = stablehlo.divide %273, %272 : tensor<512xf32>
    %275 = call @relu(%244) : (tensor<512xf32>) -> tensor<512xf32>
    %276 = stablehlo.multiply %274, %275 : tensor<512xf32>
    %277 = stablehlo.negate %260 : tensor<512xf32>
    %278 = stablehlo.exponential %277 : tensor<512xf32>
    %cst_18 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %279 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %280 = stablehlo.add %279, %278 : tensor<512xf32>
    %281 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %282 = stablehlo.divide %281, %280 : tensor<512xf32>
    %283 = stablehlo.tanh %268 : tensor<512xf32>
    %284 = stablehlo.add %282, %283 : tensor<512xf32>
    %285 = stablehlo.tanh %252 : tensor<512xf32>
    %286 = stablehlo.tanh %276 : tensor<512xf32>
    %287 = stablehlo.multiply %285, %286 : tensor<512xf32>
    %288 = stablehlo.tanh %284 : tensor<512xf32>
    %289 = stablehlo.tanh %287 : tensor<512xf32>
    %290 = stablehlo.multiply %288, %289 : tensor<512xf32>
    %291 = stablehlo.tanh %290 : tensor<512xf32>
    %292 = stablehlo.slice %arg0 [4:5, 0:512] : (tensor<5x512xf32>) -> tensor<1x512xf32>
    %293 = stablehlo.reshape %292 : (tensor<1x512xf32>) -> tensor<512xf32>
    %294 = stablehlo.dot_general %293, %arg1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %295 = stablehlo.dot_general %291, %arg9, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %296 = stablehlo.add %294, %295 : tensor<512xf32>
    %297 = stablehlo.dot_general %293, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %298 = stablehlo.dot_general %291, %arg10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %299 = stablehlo.add %297, %298 : tensor<512xf32>
    %300 = stablehlo.dot_general %293, %arg3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %301 = stablehlo.dot_general %291, %arg11, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %302 = stablehlo.add %300, %301 : tensor<512xf32>
    %303 = stablehlo.dot_general %293, %arg4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %304 = stablehlo.dot_general %291, %arg12, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %305 = stablehlo.add %303, %304 : tensor<512xf32>
    %306 = stablehlo.dot_general %293, %arg5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %307 = stablehlo.dot_general %291, %arg13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %308 = stablehlo.add %306, %307 : tensor<512xf32>
    %309 = stablehlo.dot_general %293, %arg6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %310 = stablehlo.dot_general %291, %arg14, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %311 = stablehlo.add %309, %310 : tensor<512xf32>
    %312 = stablehlo.dot_general %293, %arg7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %313 = stablehlo.dot_general %291, %arg15, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %314 = stablehlo.add %312, %313 : tensor<512xf32>
    %315 = stablehlo.dot_general %293, %arg8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %316 = stablehlo.dot_general %291, %arg16, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x512xf32>) -> tensor<512xf32>
    %317 = stablehlo.add %315, %316 : tensor<512xf32>
    %318 = call @relu(%296) : (tensor<512xf32>) -> tensor<512xf32>
    %319 = stablehlo.negate %305 : tensor<512xf32>
    %320 = stablehlo.exponential %319 : tensor<512xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %321 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %322 = stablehlo.add %321, %320 : tensor<512xf32>
    %323 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %324 = stablehlo.divide %323, %322 : tensor<512xf32>
    %325 = stablehlo.add %318, %324 : tensor<512xf32>
    %326 = stablehlo.negate %299 : tensor<512xf32>
    %327 = stablehlo.exponential %326 : tensor<512xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %328 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %329 = stablehlo.add %328, %327 : tensor<512xf32>
    %330 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %331 = stablehlo.divide %330, %329 : tensor<512xf32>
    %332 = stablehlo.tanh %302 : tensor<512xf32>
    %333 = stablehlo.add %331, %332 : tensor<512xf32>
    %334 = stablehlo.negate %308 : tensor<512xf32>
    %335 = stablehlo.exponential %334 : tensor<512xf32>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %336 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %337 = stablehlo.add %336, %335 : tensor<512xf32>
    %338 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %339 = stablehlo.divide %338, %337 : tensor<512xf32>
    %340 = stablehlo.tanh %311 : tensor<512xf32>
    %341 = stablehlo.multiply %339, %340 : tensor<512xf32>
    %342 = stablehlo.negate %314 : tensor<512xf32>
    %343 = stablehlo.exponential %342 : tensor<512xf32>
    %cst_22 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %344 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %345 = stablehlo.add %344, %343 : tensor<512xf32>
    %346 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %347 = stablehlo.divide %346, %345 : tensor<512xf32>
    %348 = call @relu(%317) : (tensor<512xf32>) -> tensor<512xf32>
    %349 = stablehlo.multiply %347, %348 : tensor<512xf32>
    %350 = stablehlo.negate %333 : tensor<512xf32>
    %351 = stablehlo.exponential %350 : tensor<512xf32>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %352 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %353 = stablehlo.add %352, %351 : tensor<512xf32>
    %354 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %355 = stablehlo.divide %354, %353 : tensor<512xf32>
    %356 = stablehlo.tanh %341 : tensor<512xf32>
    %357 = stablehlo.add %355, %356 : tensor<512xf32>
    %358 = stablehlo.tanh %325 : tensor<512xf32>
    %359 = stablehlo.tanh %349 : tensor<512xf32>
    %360 = stablehlo.multiply %358, %359 : tensor<512xf32>
    %361 = stablehlo.tanh %357 : tensor<512xf32>
    %362 = stablehlo.tanh %360 : tensor<512xf32>
    %363 = stablehlo.multiply %361, %362 : tensor<512xf32>
    %364 = stablehlo.tanh %363 : tensor<512xf32>
    return %364 : tensor<512xf32>
  }
}
