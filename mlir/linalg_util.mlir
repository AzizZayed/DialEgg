func.func private @printI64(i64)
func.func private @printF64(f64)
func.func private @printComma()
func.func private @printNewline()

// C funcs
func.func private @clock() -> i64
func.func private @putchar(i32) -> i32
llvm.func @printf(!llvm.ptr, ...) -> i32

llvm.mlir.global constant @time("%d us -> %f s")

func.func @printI64Tensor1D(%tensor : tensor<?xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %lbracket = arith.constant 91 : i32
    func.call @putchar(%lbracket) : (i32) -> i32

    %size = tensor.dim %tensor, %c0 : tensor<?xi64>
    scf.for %i = %c0 to %size step %c1 {
        %val = tensor.extract %tensor[%i] : tensor<?xi64>
        func.call @printI64(%val) : (i64) -> ()

        %size_minus_one = index.sub %size, %c1
        %not_end = index.cmp ult(%i, %size_minus_one)
        scf.if %not_end { // print comma if not last element
            func.call @printComma() : () -> ()
        }
    }

    %rbracket = arith.constant 93 : i32
    func.call @putchar(%rbracket) : (i32) -> i32

    func.return
}

func.func @printF64Tensor1D(%tensor : tensor<?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %lbracket = arith.constant 91 : i32
    func.call @putchar(%lbracket) : (i32) -> i32

    %size = tensor.dim %tensor, %c0 : tensor<?xf64>
    scf.for %i = %c0 to %size step %c1 {
        %val = tensor.extract %tensor[%i] : tensor<?xf64>
        func.call @printF64(%val) : (f64) -> ()

        %size_minus_one = index.sub %size, %c1
        %not_end = index.cmp ult(%i, %size_minus_one)
        scf.if %not_end { // print comma if not last element
            func.call @printComma() : () -> ()
        }
    }

    %rbracket = arith.constant 93 : i32
    func.call @putchar(%rbracket) : (i32) -> i32

    func.return
}

func.func @printI64Tensor2D(%tensor: tensor<?x?xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %lbracket = arith.constant 91 : i32
    func.call @putchar(%lbracket) : (i32) -> i32

    %tab = arith.constant 9 : i32

    %size0 = tensor.dim %tensor, %c0 : tensor<?x?xi64>
    %size1 = tensor.dim %tensor, %c1 : tensor<?x?xi64>

    scf.for %i = %c0 to %size0 step %c1 {
        %tensor1D = tensor.extract_slice %tensor[%i, 0][1, %size1][1, 1] : tensor<?x?xi64> to tensor<?xi64>
        
        func.call @printNewline() : () -> ()
        func.call @putchar(%tab) : (i32) -> i32
        func.call @printI64Tensor1D(%tensor1D) : (tensor<?xi64>) -> ()

        %size0_minus_one = index.sub %size0, %c1
        %not_end = index.cmp ult(%i, %size0_minus_one)
        scf.if %not_end { // print comma if not last element
            func.call @printComma() : () -> ()
        }
    }

    %size0_gt_0 = index.cmp sgt(%size0, %c0)
    scf.if %size0_gt_0 { // new line if size0 > 0
        func.call @printNewline() : () -> ()
    }

    %rbracket = arith.constant 93 : i32
    func.call @putchar(%rbracket) : (i32) -> i32

    func.return
}

func.func @printF64Tensor2D(%tensor: tensor<?x?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %lbracket = arith.constant 91 : i32
    func.call @putchar(%lbracket) : (i32) -> i32

    %tab = arith.constant 9 : i32

    %size0 = tensor.dim %tensor, %c0 : tensor<?x?xf64>
    %size1 = tensor.dim %tensor, %c1 : tensor<?x?xf64>

    scf.for %i = %c0 to %size0 step %c1 {
        %tensor1D = tensor.extract_slice %tensor[%i, 0][1, %size1][1, 1] : tensor<?x?xf64> to tensor<?xf64>
        
        func.call @printNewline() : () -> ()
        func.call @putchar(%tab) : (i32) -> i32
        func.call @printF64Tensor1D(%tensor1D) : (tensor<?xf64>) -> ()

        %size0_minus_one = index.sub %size0, %c1
        %not_end = index.cmp ult(%i, %size0_minus_one)
        scf.if %not_end { // print comma if not last element
            func.call @printComma() : () -> ()
        }
    }

    %size0_gt_0 = index.cmp sgt(%size0, %c0)
    scf.if %size0_gt_0 { // new line if size0 > 0
        func.call @printNewline() : () -> ()
    }

    %rbracket = arith.constant 93 : i32
    func.call @putchar(%rbracket) : (i32) -> i32

    func.return
}

func.func @fillRandomF64Tensor2D(%tensor: tensor<?x?xf64>) -> tensor<?x?xf64> {
    // Create a 2D tensor with random values with the linalg.fill_rng_2d op

    %seed = arith.constant 0 : i32
    %min = arith.constant -10.0 : f64
    %max = arith.constant 10.0 : f64

    %tensor_filled = linalg.fill_rng_2d ins(%min, %max, %seed : f64, f64, i32) 
                                        outs(%tensor : tensor<?x?xf64>) -> tensor<?x?xf64>

    return %tensor_filled : tensor<?x?xf64>
}

func.func @fillRandomI64Tensor2D(%tensor: tensor<?x?xi64>) -> tensor<?x?xi64> {
    // Create a 2D tensor with random values with the linalg.fill_rng_2d op

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %cols = tensor.dim %tensor, %c1 : tensor<?x?xi64>
    %rows = tensor.dim %tensor, %c0 : tensor<?x?xi64>

    %seed = arith.constant 0 : i32
    %min = arith.constant -10.0 : f64
    %max = arith.constant 10.0 : f64
    %init = tensor.empty(%rows, %cols) : tensor<?x?xf64>

    %init_filled = linalg.fill_rng_2d ins(%min, %max, %seed : f64, f64, i32) 
                                      outs(%init : tensor<?x?xf64>) -> tensor<?x?xf64>

    // Floor each value and cast to i64 with generic op
    %init_floor = linalg.floor ins(%init_filled : tensor<?x?xf64>) outs(%init : tensor<?x?xf64>) -> tensor<?x?xf64>
    %tensor_filled = linalg.generic 
        {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%init_floor : tensor<?x?xf64>) 
        outs(%tensor : tensor<?x?xi64>) {

        ^bb(%arg0: f64, %arg1: i64):
            %int_val = arith.fptosi %arg0 : f64 to i64
            linalg.yield %int_val : i64
    } -> tensor<?x?xi64>
    // %tensor_filled = arith.fptosi %init_floor : tensor<?x?xf64> to tensor<?x?xi64>

    return %tensor_filled : tensor<?x?xi64>
}

func.func @displayTime(%start: i64, %end: i64) {
    %diff = arith.subi %end, %start : i64
    %diff_f64 = arith.uitofp %diff : i64 to f64

    %million = arith.constant 1000000.0 : f64
    %diff_seconds = arith.divf %diff_f64, %million : f64

    // Format: "%f us -> %f s"
    %time_ptr = llvm.mlir.addressof @time : !llvm.ptr

    func.call @printNewline() : () -> ()
    func.call @printNewline() : () -> ()
    llvm.call @printf(%time_ptr, %diff, %diff_seconds) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64) -> i32

    func.return
}

func.func @main() -> f32 {
    // Start measuring time
    %start = func.call @clock() : () -> i64

    %x = tensor.empty() : tensor<10x3xi64>
    %x_cast = tensor.cast %x : tensor<10x3xi64> to tensor<?x?xi64>
    %x_filled = func.call @fillRandomI64Tensor2D(%x_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    func.call @printI64Tensor2D(%x_filled) : (tensor<?x?xi64>) -> ()

    // End measuring time
    %end = func.call @clock() : () -> i64
    func.call @displayTime(%start, %end) : (i64, i64) -> ()

    %c0 = arith.constant 0.0 : f32
    func.return %c0 : f32
}