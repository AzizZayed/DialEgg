func.func private @printNewline()
func.func private @printF64Tensor2D(%tensor : tensor<?x?xf64>)

func.func @main() -> i32 {
    %c2 = stablehlo.constant dense<2.0> : tensor<4x5xf64>
    %c3 = stablehlo.constant dense<3.0> : tensor<5x6xf64>

    %result = stablehlo.dot_general %c2, %c3,
      batching_dims = [] x [],
      contracting_dims = [1] x [0]
      : (tensor<4x5xf64>, tensor<5x6xf64>) -> tensor<4x6xf64>

    %res_cast = tensor.cast %result : tensor<4x6xf64> to tensor<?x?xf64>
    func.call @printF64Tensor2D(%res_cast) : (tensor<?x?xf64>) -> ()
    func.call @printNewline() : () -> ()

    %c0 = arith.constant 0 : i32
    func.return %c0 : i32
}

