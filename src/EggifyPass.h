#ifndef EGGIFY_PASS_H
#define EGGIFY_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

// TODO fix this path: we changed the output of an operation in egglog so make sure it is consistent

struct EggifyPass : public mlir::PassWrapper<EggifyPass, mlir::OperationPass<mlir::func::FuncOp>> {
    mlir::StringRef getArgument() const override { return "eggify"; }
    mlir::StringRef getDescription() const override { return "Converts MLIR operations to Egglog Op variants."; }

    void runOnOperation() override {
        mlir::func::FuncOp rootOp = getOperation();

        // Get the name of the function
        llvm::StringRef rootOpName = rootOp.getName();

        llvm::outs() << "--------------------------------\n";
        llvm::outs() << "Function: " << rootOpName << "\n";

        // Get all operations in the function
        rootOp.walk([](mlir::Operation* op) {
            mlir::OperationName opName = op->getName();
            llvm::StringRef opNameStr = opName.getStringRef();
            size_t numOperands = op->getNumOperands();
            size_t numAttributes = opName.getAttributeNames().size();

            // Print the operation name and the number of operands, results, and attributes
            // (function <opName> (i64 [Op]*numOperands [AttrPair]*numAttributes Type) Op)
            llvm::outs() << "(function " << opNameStr << " (";  // (function <opName> (
            for (size_t i = 0; i < numOperands; i++) {
                llvm::outs() << " Op";
            }
            for (size_t i = 0; i < numAttributes; i++) {
                llvm::outs() << " AttrPair";
            }
            llvm::outs() << " Type) Op)\n";
        });

        llvm::outs() << "--------------------------------\n";
    }
};

#endif  // EGGIFY_PASS_H