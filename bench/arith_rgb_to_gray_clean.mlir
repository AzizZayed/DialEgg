func.func @rgb_to_grayscale(%r: i64, %g: i64, %b: i64) -> i64 {
	%c256 = arith.constant 256 : i64
	
	%w_r = arith.constant 77 : i64
	%w_g = arith.constant 150 : i64
	%w_b = arith.constant 29 : i64
	
	%r_s = arith.muli %r, %w_r : i64
	%g_s = arith.muli %g, %w_g : i64
	%b_s = arith.muli %b, %w_b : i64

	%r_f = arith.divsi %r_s, %c256 : i64
	%g_f = arith.divsi %g_s, %c256 : i64
	%b_f = arith.divsi %b_s, %c256 : i64

	%sum = arith.addi %r_f, %g_f : i64
	%gray = arith.addi %sum, %b_f : i64

	func.return %gray : i64
}

func.func @main() -> i64 { // convert 4K RGB image to grayscale
    %val = arith.constant 100 : i64
    %image = tensor.empty() : tensor<3840x2160x3xi64>  // 3840x2160 image with 3 channels (r, g, b)
    %image_filled_0  = linalg.fill ins(%val : i64) outs(%image : tensor<3840x2160x3xi64>) -> tensor<3840x2160x3xi64>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %rows = arith.constant 3840 : index
    %cols = arith.constant 2160 : index
    %image_gray_init = tensor.empty() : tensor<3840x2160xi64>

    %image_gray = scf.for %i = %c0 to %rows step %c1 iter_args(%current_image = %image_gray_init) -> (tensor<3840x2160xi64>) {
        %row_result = scf.for %j = %c0 to %cols step %c1 iter_args(%current_row = %current_image) -> (tensor<3840x2160xi64>) {
            %r = tensor.extract %image_filled[%i, %j, %c0] : tensor<3840x2160x3xi64>
            %g = tensor.extract %image_filled[%i, %j, %c1] : tensor<3840x2160x3xi64>
            %b = tensor.extract %image_filled[%i, %j, %c2] : tensor<3840x2160x3xi64>

            %gray = func.call @rgb_to_grayscale(%r, %g, %b) : (i64, i64, i64) -> i64

            %updated_row = tensor.insert %gray into %current_row[%i, %j] : tensor<3840x2160xi64>
            scf.yield %updated_row : tensor<3840x2160xi64>
        }
        scf.yield %row_result : tensor<3840x2160xi64>
    }

    %c0_i64 = arith.constant 0 : i64
    func.return %c0_i64 : i64
}