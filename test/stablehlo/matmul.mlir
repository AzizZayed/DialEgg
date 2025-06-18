func.func private @printNewline()
func.func private @printF64Tensor1D(%tensor : tensor<?xf64>)

// (matmul 0 (matmul 0 ?X ?Y) ?Z) => (matmul 0 ?X (matmul 0 ?Y ?Z))
// func.func @matmul(%x: tensor<100x10xi64>, %y: tensor<10x150xi64>, %z: tensor<150x8xi64>) -> tensor<100x8xi64> {
//     %result = stablehlo.dot_general(%lhs, %rhs) {
//       lhs_contracting_dimensions = dense<1> : tensor<1xi64>,
//       rhs_contracting_dimensions = dense<0> : tensor<1xi64>,
//       lhs_batching_dimensions = dense<> : tensor<0xi64>,
//       rhs_batching_dimensions = dense<> : tensor<0xi64>
//     } : (tensor<10x15>, tensor<15x5>) -> tensor<10x5>
//     return %result : tensor<10x5>
// }

func.func @main() -> i32 {
    %cst = stablehlo.constant dense<2.0> : tensor<4xf64>
	%res = stablehlo.multiply %cst, %cst : tensor<4xf64>

    %res_cast = tensor.cast %res : tensor<4xf64> to tensor<?xf64>
    func.call @printF64Tensor1D(%res_cast) : (tensor<?xf64>) -> ()
    func.call @printNewline() : () -> ()

    %c0 = arith.constant 0 : i32
    func.return %c0 : i32
}

