#ifndef MATRIX_MULTIPLY_ASSOCIATE_PASS_H
#define MATRIX_MULTIPLY_ASSOCIATE_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

class MatrixMultiplyAssociateRewritePattern : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
public:
    MatrixMultiplyAssociateRewritePattern(mlir::MLIRContext* context) : OpRewritePattern<mlir::linalg::MatmulOp>(context) {}

    mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op, mlir::PatternRewriter& rewriter) const override {  // Matches either (XY)Z or X(YZ)
        // llvm::outs() << "Matching and rewriting matmul operation: " << op << "\n";

        mlir::Value lhs = op.getOperand(0);  // could be X or (XY)
        mlir::Value rhs = op.getOperand(1);  // could be (YZ) or Z
        // llvm::outs() << "lhs: " << lhs << "\n";
        // llvm::outs() << "rhs: " << rhs << "\n";

        mlir::Operation* lhsOp = lhs.getDefiningOp();  // could be nullptr
        mlir::Operation* rhsOp = rhs.getDefiningOp();  // could be nullptr

        // if (lhsOp != nullptr) llvm::outs() << "lhsOp: " << *lhsOp << "\n";
        // if (rhsOp != nullptr) llvm::outs() << "rhsOp: " << *rhsOp << "\n";

        mlir::Value x, y, z;
        bool is_x_yz = false, is_xy_z = false;
        if (lhsOp != nullptr && llvm::isa<mlir::linalg::MatmulOp>(lhsOp)) {
            is_xy_z = true;
            x = lhsOp->getOperand(0);  // a*b
            y = lhsOp->getOperand(1);  // b*c
            z = rhs;                   // c*d
        } else if (rhsOp != nullptr && llvm::isa<mlir::linalg::MatmulOp>(rhsOp)) {
            is_x_yz = true;
            x = lhs;                   // a*b
            y = rhsOp->getOperand(0);  // b*c
            z = rhsOp->getOperand(1);  // c*d
        } else {
            return mlir::failure();  // this is just a normal matmul (XY)
        }

        // llvm::outs() << "x: " << x << " | " << x.getType() << "\n";
        // llvm::outs() << "y: " << y << " | " << y.getType() << "\n";
        // llvm::outs() << "z: " << z << " | " << z.getType() << "\n";

        mlir::RankedTensorType xType = cast<mlir::RankedTensorType>(x.getType());
        mlir::RankedTensorType yType = cast<mlir::RankedTensorType>(y.getType());
        mlir::RankedTensorType zType = cast<mlir::RankedTensorType>(z.getType());
        mlir::Type opType = xType.getElementType();

        // fail if there are any dynamic dimensions
        if (xType.getNumDynamicDims() > 0 || yType.getNumDynamicDims() > 0 || zType.getNumDynamicDims() > 0) {
            return mlir::failure();
        }

        int64_t a = xType.getShape()[0];
        int64_t b = xType.getShape()[1];
        int64_t c = yType.getShape()[1];
        int64_t d = zType.getShape()[1];

        int64_t x_yzCost = b * c * d + a * b * d;
        int64_t xy_zCost = a * b * c + a * c * d;

        if (xy_zCost < x_yzCost && is_x_yz) {  // (XY)Z is better than X(YZ)
            mlir::Type xyType = mlir::RankedTensorType::get({a, c}, opType);
            mlir::Value xyInit = rewriter.create<mlir::tensor::EmptyOp>(op.getLoc(), xyType, mlir::ValueRange {});
            mlir::linalg::MatmulOp xy = rewriter.create<mlir::linalg::MatmulOp>(op.getLoc(), xyType, mlir::ValueRange {x, y}, xyInit);

            // llvm::outs() << "xy: " << xy.getResult(0) << "\n";

            mlir::Type xyzType = mlir::RankedTensorType::get({a, d}, opType);
            mlir::Value xyzInit = rewriter.create<mlir::tensor::EmptyOp>(op.getLoc(), xyzType, mlir::ValueRange {});
            mlir::linalg::MatmulOp xyz = rewriter.create<mlir::linalg::MatmulOp>(op.getLoc(), xyzType, mlir::ValueRange {xy.getResult(0), z}, xyzInit);

            // llvm::outs() << "Replacing with: " << xyz.getResult(0) << "\n";

            rewriter.replaceOp(op, xyz.getResult(0));
        } else if (x_yzCost < xy_zCost && is_xy_z) {  // X(YZ) is better than (XY)Z
            mlir::Type yzType = mlir::RankedTensorType::get({b, d}, opType);
            mlir::Value yzInit = rewriter.create<mlir::tensor::EmptyOp>(op.getLoc(), yzType, mlir::ValueRange {});
            mlir::linalg::MatmulOp yz = rewriter.create<mlir::linalg::MatmulOp>(op.getLoc(), yzType, mlir::ValueRange {y, z}, yzInit);

            // llvm::outs() << "yz: " << yz.getResult(0) << "\n";

            mlir::Type xyzType = mlir::RankedTensorType::get({a, d}, opType);
            mlir::Value xyzInit = rewriter.create<mlir::tensor::EmptyOp>(op.getLoc(), xyzType, mlir::ValueRange {});
            mlir::linalg::MatmulOp xyz = rewriter.create<mlir::linalg::MatmulOp>(op.getLoc(), xyzType, mlir::ValueRange {x, yz.getResult(0)}, xyzInit);

            // llvm::outs() << "Replacing with: " << xyz.getResult(0) << "\n";

            rewriter.replaceOp(op, xyz.getResult(0));
        }  // if equal, just keep the original

        return mlir::success();
    }
};

// Define the actual pass
class MatrixMultiplyAssociatePass : public mlir::PassWrapper<MatrixMultiplyAssociatePass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    mlir::StringRef getArgument() const override { return "matmul-associate"; }
    mlir::StringRef getDescription() const override { return "Associates matmul operations to reduce the number of multiplications."; }

    void runOnOperation() override {
        mlir::func::FuncOp func = getOperation();
        mlir::MLIRContext* context = &getContext();

        llvm::outs() << "Running MatrixMultiplyAssociateRewritePattern on function: " << func.getName() << "\n";

        mlir::RewritePatternSet patterns(context);
        patterns.add<MatrixMultiplyAssociateRewritePattern>(context);

        mlir::GreedyRewriteConfig config = mlir::GreedyRewriteConfig();
        config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
        config.setUseTopDownTraversal();
        if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), config))) {
            signalPassFailure();
        }

        llvm::outs() << "Finished running MatrixMultiplyAssociateRewritePattern on function: " << func.getName() << "\n";
    }
};

#endif  // MATRIX_MULTIPLY_ASSOCIATE_PASS_H
