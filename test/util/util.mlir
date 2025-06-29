func.func private @printI64(i64)
func.func private @printF64(f64)
func.func private @printComma()
func.func private @printNewline()

// C funcs
func.func private @clock() -> i64
func.func private @putchar(i32) -> i32

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

func.func @printI64Tensor3D(%tensor: tensor<?x?x?xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %lbracket = arith.constant 91 : i32
    %rbracket = arith.constant 93 : i32
    func.call @putchar(%lbracket) : (i32) -> i32

    %tab = arith.constant 9 : i32

    %size0 = tensor.dim %tensor, %c0 : tensor<?x?x?xi64>
    %size1 = tensor.dim %tensor, %c1 : tensor<?x?x?xi64>
    %size2 = tensor.dim %tensor, %c2 : tensor<?x?x?xi64>

    scf.for %i = %c0 to %size0 step %c1 {
        func.call @printNewline() : () -> ()
        func.call @putchar(%tab) : (i32) -> i32
        func.call @putchar(%lbracket) : (i32) -> i32

        %tensor2D = tensor.extract_slice %tensor[%i, 0, 0][1, %size1, %size2][1, 1, 1] : tensor<?x?x?xi64> to tensor<?x?xi64>
        func.call @printI64Tensor2D(%tensor2D) : (tensor<?x?xi64>) -> ()

        func.call @putchar(%rbracket) : (i32) -> i32

        %size0_minus_one = index.sub %size0, %c1
        %not_end_i = index.cmp ult(%i, %size0_minus_one)
        scf.if %not_end_i {
            func.call @printComma() : () -> ()
        }
    }

    %size0_gt_0 = index.cmp sgt(%size0, %c0)
    scf.if %size0_gt_0 {
        func.call @printNewline() : () -> ()
    }

    
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

func.func @printF32Tensor2D(%tensor: tensor<?x?xf32>) {
    %tensor_f64 = arith.extf %tensor : tensor<?x?xf32> to tensor<?x?xf64>
    func.call @printF64Tensor2D(%tensor_f64) : (tensor<?x?xf64>) -> ()

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
    %tensor_filled = arith.fptosi %init_floor : tensor<?x?xf64> to tensor<?x?xi64>

    return %tensor_filled : tensor<?x?xi64>
}

func.func @displayTime(%start: i64, %end: i64) {
    %diff = arith.subi %end, %start : i64
    %diff_f64 = arith.uitofp %diff : i64 to f64

    %million = arith.constant 1000000.0 : f64
    %diff_seconds = arith.divf %diff_f64, %million : f64

    // Format: "%f us -> %f s"
    %u = arith.constant 117 : i32
    %s = arith.constant 115 : i32
    %space = arith.constant 32 : i32
    %dash = arith.constant 45 : i32
    %gt = arith.constant 62 : i32

    func.call @printF64(%diff_f64) : (f64) -> ()
    func.call @putchar(%space) : (i32) -> i32
    func.call @putchar(%u) : (i32) -> i32
    func.call @putchar(%s) : (i32) -> i32
    func.call @putchar(%space) : (i32) -> i32
    func.call @putchar(%dash) : (i32) -> i32
    func.call @putchar(%gt) : (i32) -> i32
    func.call @putchar(%space) : (i32) -> i32
    func.call @printF64(%diff_seconds) : (f64) -> ()
    func.call @putchar(%space) : (i32) -> i32
    func.call @putchar(%s) : (i32) -> i32

    func.return
}
