#include "llvm/Support/FileSystem.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/AsmParser/AsmParser.h"

#include "stablehlo/dialect/Register.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/tests/CheckOps.h"
#include "stablehlo/tests/TestUtils.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

#include "Egglog.h"
#include "EqualitySaturationPass.h"
#include "EggifyPass.h"
#include "EgglogCustomDefs.h"

#define DEBUG_TYPE "dialegg"

std::string getMlirFile(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            return argv[i];
        }
    }
    return "";
}

int main(int argc, char** argv) {
    // Register dialects
    mlir::DialectRegistry dialectRegistry;
    mlir::registerAllDialects(dialectRegistry);
    mlir::stablehlo::registerAllDialects(dialectRegistry);

    // Register passes
    mlir::registerAllPasses();
    mlir::stablehlo::registerPassPipelines();
    mlir::stablehlo::registerPasses();
    mlir::stablehlo::registerOptimizationPasses();
    mlir::stablehlo::registerStablehloLinalgTransformsPasses();
    mlir::PassRegistration<EggifyPass>();

    std::map<std::string, AttrStringifyFunction> attrStringifiers = {
            {mlir::arith::FastMathFlagsAttr::name.str(), stringifyFastMathFlagsAttr}};
    std::map<std::string, AttrParseFunction> attrParsers = {
            {"arith_fastmath", parseFastMathFlagsAttr}};

    std::map<std::string, TypeStringifyFunction> typeStringifiers = {
            {mlir::RankedTensorType::name.str(), stringifyRankedTensorType}};
    std::map<std::string, TypeParseFunction> typeParsers = {
            {"RankedTensor", parseRankedTensorType}};

    EgglogCustomDefs funcs = {attrStringifiers, attrParsers, typeStringifiers, typeParsers};
    mlir::PassRegistration<EqualitySaturationPass>([&] {
        return std::make_unique<EqualitySaturationPass>(getMlirFile(argc, argv), funcs);
    });

    // Run the main MLIR opt
    mlir::LogicalResult result = mlir::MlirOptMain(argc, argv, "EqSat", dialectRegistry);
    return mlir::asMainReturnCode(result);
}
