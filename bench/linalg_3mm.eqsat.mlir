module {
  func.func private @printNewline()
  func.func private @clock() -> i64
  func.func private @putchar(i32) -> i32
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global external constant @time("%d us -> %f s") {addr_space = 0 : i32}
  func.func @displayTime(%arg0: i64, %arg1: i64) {
    %0 = arith.subi %arg1, %arg0 : i64
    %1 = arith.uitofp %0 : i64 to f64
    %cst = arith.constant 9.000000e+03 : f64
    %2 = arith.divf %1, %cst : f64
    %3 = llvm.mlir.addressof @time : !llvm.ptr
    %4 = llvm.call @printf(%3, %0, %2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64) -> i32
    return
  }
  func.func @blackhole(%arg0: tensor<200x10xi64>) -> tensor<200x10xi64> {
    return %arg0 : tensor<200x10xi64>
  }
  func.func @main() -> i32 {
    %0 = call @clock() : () -> i64
    %1 = tensor.empty() : tensor<200x175xi64>
    %2 = tensor.empty() : tensor<175x250xi64>
    %3 = tensor.empty() : tensor<200x250xi64>
    %4 = tensor.empty() : tensor<200x175xi64>
    %5 = tensor.empty() : tensor<175x250xi64>
    %6 = tensor.empty() : tensor<250x150xi64>
    %7 = tensor.empty() : tensor<175x150xi64>
    %8 = linalg.matmul ins(%5, %6 : tensor<175x250xi64>, tensor<250x150xi64>) outs(%7 : tensor<175x150xi64>) -> tensor<175x150xi64>
    %9 = tensor.empty() : tensor<200x150xi64>
    %10 = tensor.empty() : tensor<200x175xi64>
    %11 = tensor.empty() : tensor<175x250xi64>
    %12 = tensor.empty() : tensor<250x150xi64>
    %13 = tensor.empty() : tensor<150x10xi64>
    %14 = tensor.empty() : tensor<250x10xi64>
    %15 = linalg.matmul ins(%12, %13 : tensor<250x150xi64>, tensor<150x10xi64>) outs(%14 : tensor<250x10xi64>) -> tensor<250x10xi64>
    %16 = tensor.empty() : tensor<175x10xi64>
    %17 = linalg.matmul ins(%11, %15 : tensor<175x250xi64>, tensor<250x10xi64>) outs(%16 : tensor<175x10xi64>) -> tensor<175x10xi64>
    %18 = tensor.empty() : tensor<200x10xi64>
    %19 = linalg.matmul ins(%10, %17 : tensor<200x175xi64>, tensor<175x10xi64>) outs(%18 : tensor<200x10xi64>) -> tensor<200x10xi64>
    %20 = call @clock() : () -> i64
    call @displayTime(%0, %20) : (i64, i64) -> ()
    %21 = call @blackhole(%19) : (tensor<200x10xi64>) -> tensor<200x10xi64>
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
