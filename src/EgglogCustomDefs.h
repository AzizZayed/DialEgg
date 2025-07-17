#ifndef EGGLOG_CUSTOM_DEFS_H
#define EGGLOG_CUSTOM_DEFS_H

#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/AsmParser/AsmParser.h"

#include "Egglog.h"

/** 
 * Parse the attribute (function arith_fastmath (FastMathFlags) Attr)
 * Where (datatype FastMathFlags (none) (reassoc) (nnan) ...)
 */
mlir::Attribute parseFastMathFlagsAttr(const std::vector<std::string_view>& split, Egglog& egglog) {
    std::string_view attrType = split[0];
    assert(attrType == "arith_fastmath");

    std::string_view flag = split[1];
    if (flag.front() == '(' && flag.back() == ')') {
        flag = flag.substr(1, flag.size() - 2);
    }

    std::string strAttr = "#arith.fastmath<" + std::string(flag) + ">";
    mlir::Attribute parsedAttr = mlir::parseAttribute(strAttr, &egglog.context);

    // dump
    // llvm::outs() << "Parsing mlir::arith::FastMathFlagsAttr: " << strAttr << "\n";
    // llvm::outs() << "Parsed mlir::arith::FastMathFlagsAttr: " << parsedAttr << "\n";

    return parsedAttr;
}

/**
 * Serialize the attribute (function arith_fastmath (FastMathFlags) Attr)
 * Where (datatype FastMathFlags (none) (reassoc) (nnan) ...)
 */
std::vector<std::string> stringifyFastMathFlagsAttr(mlir::Attribute attr, Egglog& egglog) {
    std::vector<std::string> split;

    split.push_back("arith_fastmath");

    mlir::arith::FastMathFlagsAttr fastMathAttr = mlir::cast<mlir::arith::FastMathFlagsAttr>(attr);
    mlir::arith::FastMathFlags flags = fastMathAttr.getValue();

    split.push_back("(" + mlir::arith::stringifyFastMathFlags(flags) + ")");

    // dump
    // llvm::outs() << "Stringified mlir::arith::FastMathFlagsAttr: ";
    // for (const std::string& s: split) {
    //     llvm::outs() << s << " , ";
    // }
    // llvm::outs() << "\n";

    return split;
}

/** Parse the type (function RankedTensor (IntVec Type) Type) */
mlir::Type parseRankedTensorType(const std::vector<std::string_view>& split, Egglog& egglog) {
    std::string_view attrType = split[0];
    assert(attrType == "RankedTensor");

    std::string_view dimVec = split[1];
    std::string_view type = split[2];

    // parse dimvec, form (vec-of <N1> <N2> ... <Nn>)
    std::vector<std::string_view> dimVecSplit = Egglog::splitExpression(dimVec);
    std::vector<int64_t> dims;
    for (unsigned int i = 1; i < dimVecSplit.size(); i++) {
        dims.push_back(std::stoll(std::string(dimVecSplit[i])));
    }

    mlir::Type parsedType = egglog.parseType(type);  // parse type
    return mlir::RankedTensorType::get(dims, parsedType);
}

/** Serialize the type (function RankedTensor (IntVec Type) Type) */
std::vector<std::string> stringifyRankedTensorType(mlir::Type type, Egglog& egglog) {
    std::vector<std::string> split;
    split.push_back("RankedTensor");

    mlir::RankedTensorType tensorType = mlir::cast<mlir::RankedTensorType>(type);
    llvm::ArrayRef<int64_t> shape = tensorType.getShape();
    mlir::Type elementType = tensorType.getElementType();

    // serialize dimvec
    std::string dimVec = "(vec-of";
    for (int64_t dim: shape) {
        dimVec += " " + std::to_string(dim);
    }
    dimVec += ")";

    split.push_back(dimVec);
    split.push_back(egglog.eggifyType(elementType));

    return split;
}

#endif  // EGGLOG_CUSTOM_DEFS_H
