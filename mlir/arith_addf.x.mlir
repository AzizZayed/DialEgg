module {
  func.func private @printF64(f64)
  func.func private @printNewline()
  func.func @main() -> f32 {
    %cst = arith.constant 1.500000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant 3.500000e+00 : f64
    call @printF64(%cst_1) : (f64) -> ()
    call @printNewline() : () -> ()
    %0 = arith.subf %cst_0, %cst : f64
    call @printF64(%0) : (f64) -> ()
    call @printNewline() : () -> ()
    %cst_2 = arith.constant 0.000000e+00 : f32
    return %cst_2 : f32
  }
}
