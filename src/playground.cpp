#include "llvm/Support/FileSystem.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/AsmParser/AsmParser.h"

#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace llvm;

int main() {
    DialectRegistry dialectRegistry;
    // registerAllDialects(dialectRegistry);
    stablehlo::registerAllDialects(dialectRegistry);

    MLIRContext context(dialectRegistry);
    context.loadAllAvailableDialects();

    // Get loaded dialectsAdd commentMore actions
    outs() << "Loaded dialects: ";
    for (const Dialect* dialect: context.getLoadedDialects()) {
        outs() << dialect->getNamespace() << " ";
    }
    outs() << "\n";

    //static DotAlgorithmAttr get(MLIRContext, Type, Type, Type, int64_t, int64_t, int64_t, bool);
    stablehlo::DotAlgorithmAttr dotAttr = stablehlo::DotAlgorithmAttr::get(&context,
                                                                           FloatTF32Type::get(&context),
                                                                           FloatTF32Type::get(&context),
                                                                           Float32Type::get(&context),
                                                                           1, 1, 1, false);
    outs() << "DotAlgorithmAttr: " << dotAttr << "\n";

    // static DotDimensionNumbersAttr get(MLIRContext, ArrayRef<int64_t>, ArrayRef<int64_t>, ArrayRef<int64_t>, ArrayRef<int64_t>);
    stablehlo::DotDimensionNumbersAttr dotDimAttr = stablehlo::DotDimensionNumbersAttr::get(&context, {0}, {0}, {1}, {0});
    outs() << "DotDimensionNumbersAttr: " << dotDimAttr << "\n";

    // static PrecisionAttr get(::mlir::MLIRContext *context, ::mlir::stablehlo::Precision value);
    
    
    return 0;
}