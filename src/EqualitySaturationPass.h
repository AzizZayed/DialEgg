#ifndef EQUALITYSATURATIONPASS_H
#define EQUALITYSATURATIONPASS_H

#include <set>
#include <map>
#include <unordered_map>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "Egglog.h"
#include "Utils.h"

struct EqualitySaturationPass : public mlir::PassWrapper<EqualitySaturationPass, mlir::OperationPass<mlir::ModuleOp>> {
    std::string mlirFilePath;
    std::string eggFilePath;

    std::string egglogExtractedFilename = "egglog-extract.txt";
    std::string egglogLogFilename = "egglog-log.txt";

    EgglogCustomDefs customFunctions;

    std::map<std::string, EgglogOpDef, std::less<>> supportedOps;
    std::set<std::string> supportedDialects;

    double mlirToEgglogTime = 0.0;
    double egglogExecTime = 0.0;
    double egglogToMlirTime = 0.0;

    EqualitySaturationPass(const std::string&, const EgglogCustomDefs&);

    mlir::StringRef getArgument() const override { return "eq-sat"; }
    mlir::StringRef getDescription() const override { return "Performs equality saturation on each block in the given file. The language definition is egglog."; }

    llvm::LogicalResult initialize(mlir::MLIRContext*) override;
    void runOnOperation() override;
    void runOnFunction(mlir::func::FuncOp&);
    void runOnBlock(mlir::Block&, const std::string&);
    void runEgglog(const std::vector<EggifiedOp*>&, const std::string&);
};

#endif  // EQUALITYSATURATIONPASS_H
