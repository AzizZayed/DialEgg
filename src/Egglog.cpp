#include <regex>
#include <chrono>

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "Egglog.h"
#include "EqualitySaturationPass.h"
#include "Utils.h"

#define DEBUG_TYPE "dialegg"

EgglogOpDef EgglogOpDef::parseOpFunction(const std::string& opStr) {
    std::string newOpStr = Egglog::removeComment(opStr);
    std::vector<std::string> split = Egglog::splitExpression(newOpStr);
    assert(split.size() > 0);
    assert(split[0] == "function");

    // check if the last element is "Op", otherwise the cost is the last element
    size_t cost = 1;
    if (split.back() != "Op") {
        cost = std::stoll(split.back());
        assert(split[split.size() - 3] == "Op");
    } else {
        assert(split.back() == "Op");
    }

    std::string fullName = split[1]; // dialect_name_name_name_version or dialect_name_name_name

    size_t firstUnderscore = fullName.find_first_of('_');
    std::string dialect = fullName.substr(0, firstUnderscore);

    std::string version = "";
    std::string name = fullName.substr(firstUnderscore + 1);
    if (isdigit(fullName.back())) {
        size_t lastUnderscore = fullName.find_last_of('_');
        version = fullName.substr(lastUnderscore + 1);
        name = fullName.substr(firstUnderscore + 1, lastUnderscore - firstUnderscore - 1);
    }

    LLVM_DEBUG(llvm::dbgs() << "Parsing EgglogOpDef: " << newOpStr << "\n"
                << "Full Name: " << fullName << "\n"
                << "Dialect: " << dialect << ", Name: " << name << ", Version: " << version << "\n");

    std::vector<std::string> args = Egglog::splitExpression(split[2]);

    size_t nOperands = 0;
    size_t nAttributes = 0;
    size_t nRegions = 0;
    size_t nResults = 0;
    for (const std::string& arg: args) {
        if (arg == "Op") {
            nOperands++;
        } else if (arg == "AttrPair") {
            nAttributes++;
        } else if (arg == "Region") {
            nRegions++;
        } else if (arg == "Type") {
            nResults++;
        }
    }

    return EgglogOpDef {
            .str = newOpStr,
            .fullName = fullName,
            .dialect = dialect,
            .name = name,
            .version = version,
            .args = args,
            .nOperands = nOperands,
            .nAttributes = nAttributes,
            .nRegions = nRegions,
            .nResults = nResults,
            .cost = cost,
    };
}

bool EgglogOpDef::isOpFunction(const std::string& opStr) {
    std::string newOpStr = Egglog::removeComment(opStr);
    if (newOpStr.empty()) {
        return false;
    } else if (newOpStr.front() != '(' || newOpStr.back() != ')') {
        return false;
    }

    std::vector<std::string> split = Egglog::splitExpression(newOpStr);
    return split.size() > 1 && split[0] == "function" && (split.back() == "Op" || split[split.size() - 3] == "Op");
}

std::string Egglog::removeComment(const std::string& str) {
    // remove everything after the first semicolon
    size_t semicolon = str.find_first_of(';');
    if (semicolon != std::string::npos) {
        return str.substr(0, semicolon);
    }
    return str;
}

std::vector<std::string> Egglog::splitExpression(std::string opStr) {
    // The expression must be surrounded by parentheses
    if (opStr.front() != '(' || opStr.back() != ')') {
        llvm::outs() << "Invalid expression: " << opStr << "\n";
        exit(1);
    }

    // split by whitespace that is not inside parentheses, square brackets, or quotes
    std::vector<std::string> result;
    std::stringstream ss;

    int openParentheses = 0;
    bool inQuotes = false;
    for (size_t i = 1; i < opStr.size() - 1; i++) {
        char c = opStr[i];

        if (c == '(' || c == '[') {
            openParentheses++;
        } else if (c == ')' || c == ']') {
            openParentheses--;
        }

        if (c == '"' && opStr[i - 1] != '\\') {
            inQuotes = !inQuotes;
        }

        if (iswspace(c) && openParentheses == 0 && !inQuotes && !isBlank(ss.str())) {
            result.push_back(ss.str());
            ss.str("");
        } else {
            ss << c;
        }
    }

    if (!isBlank(ss.str())) {
        result.push_back(ss.str());
    }

    return result;
}

/** Parses the given type string into an MLIR type Form (F16) or (Complex <type>) */
mlir::Type Egglog::parseType(std::string typeStr) {
    std::vector<std::string> split = splitExpression(typeStr);

    std::string type = split[0];
    if (type == "F16") {
        return mlir::Float16Type::get(&context);
    } else if (type == "F32") {
        return mlir::Float32Type::get(&context);
    } else if (type == "F64") {
        return mlir::Float64Type::get(&context);
    } else if (type == "F80") {
        return mlir::Float80Type::get(&context);
    } else if (type == "F128") {
        return mlir::Float128Type::get(&context);
    } else if (type == "TF32") {
        return mlir::FloatTF32Type::get(&context);
    } else if (type == "I1") {
        return mlir::IntegerType::get(&context, 1);
    } else if (type == "I8") {
        return mlir::IntegerType::get(&context, 8);
    } else if (type == "I16") {
        return mlir::IntegerType::get(&context, 16);
    } else if (type == "I32") {
        return mlir::IntegerType::get(&context, 32);
    } else if (type == "I64") {
        return mlir::IntegerType::get(&context, 64);
    } else if (type == "Int") {
        return mlir::IntegerType::get(&context, std::stoll(split[1]), mlir::IntegerType::SignednessSemantics::Signless);
    } else if (type == "SInt") {
        return mlir::IntegerType::get(&context, std::stoll(split[1]), mlir::IntegerType::SignednessSemantics::Signed);
    } else if (type == "UInt") {
        return mlir::IntegerType::get(&context, std::stoll(split[1]), mlir::IntegerType::SignednessSemantics::Unsigned);
    } else if (type == "OtherInt") {
        return mlir::parseType(unwrap(split[1], '"'), &context);
    } else if (type == "Index") {
        return mlir::IndexType::get(&context);
    } else if (type == "None") {
        return mlir::NoneType::get(&context);
    } else if (type == "Complex") {
        mlir::Type elementType = parseType(split[1]);
        return mlir::ComplexType::get(elementType);
    } else if (type == "Tuple") {  // (Tuple (vec-of <type1> <type2> ... <typeN>))
        std::string typeVec = split[1];
        std::vector<mlir::Type> types;

        std::vector<std::string> typeSplit = splitExpression(typeVec);
        for (size_t i = 1; i < typeSplit.size(); i++) {
            types.push_back(parseType(typeSplit[i]));
        }

        return mlir::TupleType::get(&context, types);
    } else if (type == "Vector") {
        std::string dimVec = split[1];
        std::string elementTypeStr = split[2];

        std::vector<int64_t> dims;
        std::vector<std::string> dimVecSplit = splitExpression(dimVec);
        for (size_t i = 1; i < dimVecSplit.size(); i++) {
            dims.push_back(std::stoll(dimVecSplit[i]));
        }

        mlir::Type elementType = parseType(elementTypeStr);
        return mlir::VectorType::get(dims, elementType);
    } else if (type == "RankedTensor") {
        std::string dimVec = split[1];
        std::string elementTypeStr = split[2];

        std::vector<int64_t> dims;
        std::vector<std::string> dimVecSplit = splitExpression(dimVec);
        for (size_t i = 1; i < dimVecSplit.size(); i++) {
            dims.push_back(std::stoll(dimVecSplit[i]));
        }

        mlir::Type elementType = parseType(elementTypeStr);
        return mlir::RankedTensorType::get(dims, elementType);
    } else if (type == "UnrankedTensor") {
        std::string elementTypeStr = split[1];
        mlir::Type elementType = parseType(elementTypeStr);
        return mlir::UnrankedTensorType::get(elementType);
    } else if (type == "OpaqueType") {
        return mlir::parseType(unwrap(split[1], '"'), &context);
    } else if (type == "Function") {
        std::string inputTypesStr = split[1];
        std::string resultTypesStr = split[2];

        std::vector<mlir::Type> inputTypes;
        std::vector<mlir::Type> resultTypes;

        std::vector<std::string> inputTypesSplit = splitExpression(inputTypesStr);
        for (size_t i = 1; i < inputTypesSplit.size(); i++) {
            inputTypes.push_back(parseType(inputTypesSplit[i]));
        }

        std::vector<std::string> resultTypesSplit = splitExpression(resultTypesStr);
        for (size_t i = 1; i < resultTypesSplit.size(); i++) {
            resultTypes.push_back(parseType(resultTypesSplit[i]));
        }

        return mlir::FunctionType::get(&context, inputTypes, resultTypes);
    } else if (egglogCustom.typeParsers.find(type) != egglogCustom.typeParsers.end()) {
        TypeParseFunction parseFunc = egglogCustom.typeParsers.at(type);
        return parseFunc(split, *this);
    } else {
        llvm::outs() << "Unsupported type: " << type << "\n";
        exit(1);
    }
}

std::string Egglog::eggifyType(mlir::Type type) {
    std::string egglogCode;
    llvm::raw_string_ostream ss(egglogCode);

    std::string typeStr = type.getAbstractType().getName().str();
    if (type.isF16()) {
        ss << "(F16)";
    } else if (type.isF32()) {
        ss << "(F32)";
    } else if (type.isF64()) {
        ss << "(F64)";
    } else if (type.isF80()) {
        ss << "(F80)";
    } else if (type.isF128()) {
        ss << "(F128)";
    } else if (type.isTF32()) {
        ss << "(TF32)";
    } else if (type.isInteger(1)) {
        ss << "(I1)";
    } else if (type.isInteger(4)) {
        ss << "(I4)";
    } else if (type.isInteger(8)) {
        ss << "(I8)";
    } else if (type.isInteger(16)) {
        ss << "(I16)";
    } else if (type.isInteger(32)) {
        ss << "(I32)";
    } else if (type.isInteger(64)) {
        ss << "(I64)";
    } else if (isa<mlir::IntegerType>(type)) {
        mlir::IntegerType intType = cast<mlir::IntegerType>(type);
        size_t width = intType.getWidth();

        if (intType.isSignless()) {
            ss << "(Int" << width << ")";
        } else if (intType.isSigned()) {
            ss << "(SInt" << width << ")";
        } else if (intType.isUnsigned()) {
            ss << "(UInt" << width << ")";
        } else {
            ss << "(OtherInt \"" << type << "\")";
        }
    } else if (type.isIndex()) {
        ss << "(Index)";
    } else if (isa<mlir::NoneType>(type)) {
        ss << "(None)";
    } else if (isa<mlir::ComplexType>(type)) {
        ss << "(Complex " << cast<mlir::ComplexType>(type).getElementType() << ")";
    } else if (isa<mlir::TupleType>(type)) {
        mlir::TupleType tupleType = cast<mlir::TupleType>(type);
        ss << "(TupleType (";
        for (mlir::Type element: tupleType.getTypes()) {
            ss << " " << eggifyType(element);
        }
        ss << "))";
    } else if (isa<mlir::VectorType>(type)) {
        mlir::VectorType vectorType = cast<mlir::VectorType>(type);
        ss << "(Vector (vec-of";
        for (int64_t dim: vectorType.getShape()) {
            ss << " " << dim;
        }
        ss << ") " << eggifyType(vectorType.getElementType()) << ")";
    } else if (isa<mlir::RankedTensorType>(type)) {
        mlir::RankedTensorType tensorType = cast<mlir::RankedTensorType>(type);
        llvm::ArrayRef<int64_t> shape = tensorType.getShape();
        mlir::Type elementType = tensorType.getElementType();

        ss << "(RankedTensor (vec-of";
        for (int64_t dim: shape) {
            ss << " " << dim;
        }
        ss << ") " << eggifyType(elementType) << ")";
    } else if (isa<mlir::UnrankedTensorType>(type)) {
        mlir::UnrankedTensorType tensorType = cast<mlir::UnrankedTensorType>(type);
        mlir::Type elementType = tensorType.getElementType();

        ss << "(UnrankedTensor " << eggifyType(elementType) << ")";
    } else if (egglogCustom.typeStringifiers.find(typeStr) != egglogCustom.typeStringifiers.end()) {  // custom type by user
        TypeStringifyFunction stringifyFunc = egglogCustom.typeStringifiers.at(typeStr);
        std::vector<std::string> split = stringifyFunc(type, *this);
        assert(split.size() > 0);

        ss << "(" << split[0];
        for (size_t i = 1; i < split.size(); i++) {
            ss << " " << split[i];
        }
        ss << ")";
    } else if (isa<mlir::FunctionType>(type)) {
        mlir::FunctionType functionType = cast<mlir::FunctionType>(type);
        llvm::ArrayRef<mlir::Type> inputTypes = functionType.getInputs();
        llvm::ArrayRef<mlir::Type> resultTypes = functionType.getResults();

        ss << "(Function (vec-of ";
        for (mlir::Type inputType: inputTypes) {
            ss << eggifyType(inputType) << " ";
        }
        ss << ") (vec-of ";
        for (mlir::Type resultType: resultTypes) {
            ss << eggifyType(resultType) << " ";
        }
        ss << "))";
    } else {
        ss << "(OpaqueType \"" << type << "\" \"" << typeStr << "\")";
    }

    ss.flush();
    return egglogCode;
}

/** Parses the given attribute string into an MLIR named attribute. Form (NamedAttr "<name>" <attr>) */
mlir::NamedAttribute Egglog::parseNamedAttribute(const std::string& attrStr) {
    std::vector<std::string> split = splitExpression(attrStr);

    mlir::StringAttr attrName = mlir::StringAttr::get(&context, unwrap(split[1], '"'));
    mlir::Attribute attr = parseAttribute(split[2]);
    return mlir::NamedAttribute(attrName, attr);
}
std::string Egglog::eggifyNamedAttribute(mlir::NamedAttribute namedAttr) {
    // (NamedAttr "<name>" <attr>)

    std::stringstream ss;

    ss << "(NamedAttr \"" << namedAttr.getName().str() << "\" ";  // name
    ss << eggifyAttribute(namedAttr.getValue()) << ")";           // value

    return ss.str();
}

/** Parses the given attribute string into an MLIR attribute. Form (<type> <arg1> <arg2> ... <argN>) */
mlir::Attribute Egglog::parseAttribute(const std::string& attrStr) {
    std::vector<std::string> split = splitExpression(attrStr);

    std::string attrType = split[0];
    if (attrType == "IntegerAttr") {
        return mlir::IntegerAttr::get(parseType(split[2]), std::stoll(split[1]));
    } else if (attrType == "FloatAttr") {
        return mlir::FloatAttr::get(parseType(split[2]), std::stod(split[1]));
    } else if (attrType == "StringAttr") {
        return mlir::parseAttribute(split[1], &context);
    } else if (attrType == "UnitAttr") {
        return mlir::UnitAttr::get(&context);
    } else if (attrType == "TypeAttr") {
        return mlir::TypeAttr::get(parseType(split[1]));
    } else if (attrType == "ArrayAttr") {  // (ArrayAttr (vec-of <attr1> <attr2> ... <attrN>))
        std::vector<mlir::Attribute> elements;
        std::vector<std::string> arraySplit = splitExpression(split[1]);
        for (size_t i = 1; i < arraySplit.size(); i++) {
            elements.push_back(parseAttribute(arraySplit[i]));
        }
        return mlir::ArrayAttr::get(&context, elements);
    } else if (attrType == "DenseIntArrayAttr") {  // (DenseIntArrayAttr N (vec-of <int1> <int2> ... <intN>) <type>)
        int64_t size = std::stoll(split[1]);
        std::vector<int64_t> values;
        std::vector<std::string> arraySplit = splitExpression(split[2]);
        for (size_t i = 1; i < arraySplit.size(); i++) {
            values.push_back(std::stoll(arraySplit[i]));
        }
        mlir::Type type = parseType(split[3]);
        size_t byteWidth = type.getIntOrFloatBitWidth() / 8;
        return mlir::DenseArrayAttr::get(&context, type, size, llvm::ArrayRef<char>(reinterpret_cast<const char*>(values.data()), values.size() * byteWidth));
    } else if (attrType == "DenseFloatArrayAttr") {  // (DenseFloatArrayAttr N (vec-of <float1> <float2> ... <floatN>) <type>)
        int64_t size = std::stoll(split[1]);
        std::vector<double> values;
        std::vector<std::string> arraySplit = splitExpression(split[2]);
        for (size_t i = 1; i < arraySplit.size(); i++) {
            values.push_back(std::stod(arraySplit[i]));
        }
        mlir::Type type = parseType(split[3]);
        return mlir::DenseArrayAttr::get(&context, type, size, llvm::ArrayRef<char>(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double)));
    } else if (attrType == "DenseIntElementsAttr") {  // (DenseIntElementsAttr (vec-of <int1> <int2> ... <intN>) <type>)
        std::vector<int64_t> values;
        std::vector<std::string> arraySplit = splitExpression(split[1]);
        for (size_t i = 1; i < arraySplit.size(); i++) {
            values.push_back(std::stoll(arraySplit[i]));
        }
        mlir::Type type = parseType(split[2]);
        mlir::ShapedType shapedType = cast<mlir::ShapedType>(type);
        return mlir::DenseIntElementsAttr::get(shapedType, values);
    } else if (attrType == "DenseFPElementsAttr") {  // (DenseFloatElementsAttr (vec-of <float1> <float2> ... <floatN>) <type>)
        std::vector<double> values;
        std::vector<std::string> arraySplit = splitExpression(split[1]);
        for (size_t i = 1; i < arraySplit.size(); i++) {
            values.push_back(std::stod(arraySplit[i]));
        }
        mlir::Type type = parseType(split[2]);
        mlir::ShapedType shapedType = cast<mlir::ShapedType>(type);
        return mlir::DenseFPElementsAttr::get(shapedType, values);
    } else if (attrType == "SymbolRefAttr") {  // (SymbolRefAttr "<name>")
        return mlir::SymbolRefAttr::get(&context, unwrap(split[1], '"'));
    } else if (attrType == "PrecisionAttr") {  // (PrecisionAttr (DEFAULT | HIGH | HIGHEST))
        std::string precisionStr = unwrap(split[1], '"');
        mlir::stablehlo::Precision precision = mlir::stablehlo::symbolizePrecision(precisionStr).value_or(mlir::stablehlo::Precision::DEFAULT);
        return mlir::stablehlo::PrecisionAttr::get(&context, precision);
    } else if (attrType == "DotAlgorithmAttr") {  // (DotAlgorithmAttr <type1> <type2> <type3> <int1> <int2> <int3> <bool>)
        mlir::Type type1 = parseType(split[1]);
        mlir::Type type2 = parseType(split[2]);
        mlir::Type type3 = parseType(split[3]);
        int64_t int1 = std::stoll(split[4]);
        int64_t int2 = std::stoll(split[5]);
        int64_t int3 = std::stoll(split[6]);
        return mlir::stablehlo::DotAlgorithmAttr::get(&context, type1, type2, type3, int1, int2, int3, split[7] == "true");
    } else if (attrType == "DotDimensionNumbersAttr") { // (DotDimensionNumbersAttr <intvec> <intvec> <intvec> <intvec>)
        std::vector<int64_t> lhsBatchingDimensions;
        std::vector<int64_t> rhsBatchingDimensions;
        std::vector<int64_t> lhsContractingDimensions;
        std::vector<int64_t> rhsContractingDimensions;

        std::vector<std::string> lhsBatchingSplit = splitExpression(split[1]);
        for (size_t i = 1; i < lhsBatchingSplit.size(); i++) {
            lhsBatchingDimensions.push_back(std::stoll(lhsBatchingSplit[i]));
        }

        std::vector<std::string> rhsBatchingSplit = splitExpression(split[2]);
        for (size_t i = 1; i < rhsBatchingSplit.size(); i++) {
            rhsBatchingDimensions.push_back(std::stoll(rhsBatchingSplit[i]));
        }

        std::vector<std::string> lhsContractingSplit = splitExpression(split[3]);
        for (size_t i = 1; i < lhsContractingSplit.size(); i++) {
            lhsContractingDimensions.push_back(std::stoll(lhsContractingSplit[i]));
        }

        std::vector<std::string> rhsContractingSplit = splitExpression(split[4]);
        for (size_t i = 1; i < rhsContractingSplit.size(); i++) {
            rhsContractingDimensions.push_back(std::stoll(rhsContractingSplit[i]));
        }

        return mlir::stablehlo::DotDimensionNumbersAttr::get(&context, lhsBatchingDimensions, rhsBatchingDimensions, lhsContractingDimensions, rhsContractingDimensions);
    } else if (attrType == "OpaqueAttr") {  // TODO add all remaining builtin attrs (check below functions)
        return mlir::parseAttribute(unwrap(split[1], '"'), &context);
    } else if (egglogCustom.attrParsers.find(attrType) != egglogCustom.attrParsers.end()) {
        AttrParseFunction parseFunc = egglogCustom.attrParsers.at(attrType);
        return parseFunc(split, *this);
    } else {
        llvm::outs() << "Unsupported attribute type: " << attrType << "\n";
        exit(1);
    }
}

std::string Egglog::eggifyAttribute(mlir::Attribute attr) {
    std::string egglogCode;
    llvm::raw_string_ostream ss(egglogCode);

    mlir::TypeID typeId = attr.getTypeID();
    std::string typeName = attr.getAbstractAttribute().getName().str();
    if (typeId == mlir::TypeID::get<mlir::IntegerAttr>()) {  // (IntegerAttr <int> <type>)
        mlir::IntegerAttr integerAttr = cast<mlir::IntegerAttr>(attr);
        int64_t value = integerAttr.getInt();
        ss << "(IntegerAttr " << value << " " << eggifyType(integerAttr.getType()) << ")";

    } else if (typeId == mlir::TypeID::get<mlir::FloatAttr>()) {  // (FloatAttr <float> <type>)
        mlir::FloatAttr floatAttr = cast<mlir::FloatAttr>(attr);
        double value = floatAttr.getValueAsDouble();
        ss << "(FloatAttr " << doubleToString(value) << " " << eggifyType(floatAttr.getType()) << ")";

    } else if (typeId == mlir::TypeID::get<mlir::StringAttr>()) {  // (StringAttr "<string>">)
        mlir::StringAttr stringAttr = cast<mlir::StringAttr>(attr);
        std::string value = stringAttr.getValue().str();
        ss << "(StringAttr \"" << value << "\")";

    } else if (typeId == mlir::TypeID::get<mlir::ArrayAttr>()) {  // (ArrayAttr (vec-of <attr1> <attr2> ... <attrN>))
        mlir::ArrayAttr arrayAttr = cast<mlir::ArrayAttr>(attr);
        ss << "(ArrayAttr (";

        if (arrayAttr.empty()) {
            ss << "vec-empty";
        } else {
            ss << "vec-of";
            for (mlir::Attribute element: arrayAttr) {
                ss << " " << eggifyAttribute(element);
            }
        }
        ss << "))";

    } else if (typeId == mlir::TypeID::get<mlir::TypeAttr>()) {  // (TypeAttr "<type>")
        mlir::TypeAttr typeAttr = cast<mlir::TypeAttr>(attr);
        mlir::Type type = typeAttr.getValue();
        ss << "(TypeAttr " << eggifyType(type) << ")";

    } else if (typeId == mlir::TypeID::get<mlir::UnitAttr>()) {  // (UnitAttr)
        ss << "(UnitAttr)";

    } else if (typeId == mlir::TypeID::get<mlir::DenseArrayAttr>()) {
        mlir::DenseArrayAttr denseArrayAttr = cast<mlir::DenseArrayAttr>(attr);
        llvm::ArrayRef<char> rawData = denseArrayAttr.getRawData();
        mlir::Type elementType = denseArrayAttr.getElementType();
        size_t size = denseArrayAttr.size();

        // if Int type, then it is a DenseIntArrayAttr
        // if Float type, then it is a DenseFloatArrayAttr

        if (isa<mlir::IntegerType>(elementType)) {
            llvm::ArrayRef<int64_t> values(reinterpret_cast<const int64_t*>(rawData.data()), size);
            ss << "(DenseIntArrayAttr " << size << " (vec-of";
            for (int64_t value: values) {
                ss << " " << value;
            }
            ss << ") " << eggifyType(elementType) << ")";
        } else if (isa<mlir::FloatType>(elementType)) {
            llvm::ArrayRef<double> values(reinterpret_cast<const double*>(rawData.data()), size);
            ss << "(DenseFloatArrayAttr " << size << " (vec-of";
            for (double value: values) {
                ss << " " << value;
            }
            ss << ") " << eggifyType(elementType) << ")";
        }

    } else if (typeId == mlir::TypeID::get<mlir::DenseIntOrFPElementsAttr>()) {
        mlir::DenseIntOrFPElementsAttr denseIntOrFPAttr = cast<mlir::DenseIntOrFPElementsAttr>(attr);
        mlir::Type elementType = denseIntOrFPAttr.getElementType();
        mlir::ShapedType shapedType = denseIntOrFPAttr.getType();

        if (isa<mlir::IntegerType>(elementType)) {
            auto values = denseIntOrFPAttr.getValues<int64_t>();  // TODO this assumes always 8 bytes per int (WRONG)
            ss << "(DenseIntElementsAttr (vec-of";
            for (int64_t value: values) {
                ss << " " << value;
            }
            ss << ") " << eggifyType(shapedType) << ")";
        } else if (isa<mlir::FloatType>(elementType)) {
            auto values = denseIntOrFPAttr.getValues<double>();  // TODO this assumes always 8 bytes per double (WRONG)
            ss << "(DenseFPElementsAttr (vec-of";
            for (double value: values) {
                ss << " " << value;
            }
            ss << ") " << eggifyType(shapedType) << ")";
        }

    } else if (typeId == mlir::TypeID::get<mlir::SymbolRefAttr>()) {  // (SymbolRefAttr "<name>")
        mlir::SymbolRefAttr symbolRefAttr = cast<mlir::SymbolRefAttr>(attr);
        std::string value = symbolRefAttr.getRootReference().str();
        ss << "(SymbolRefAttr \"" << value << "\")";
    } else if (typeId == mlir::TypeID::get<mlir::stablehlo::PrecisionAttr>()) {  // (PrecisionAttr <type>)
        mlir::stablehlo::PrecisionAttr precisionAttr = cast<mlir::stablehlo::PrecisionAttr>(attr);
        ss << "(PrecisionAttr (" << mlir::stablehlo::stringifyPrecision(precisionAttr.getValue()) << "))";
    } else if (typeId == mlir::TypeID::get<mlir::stablehlo::DotAlgorithmAttr>()) {
        mlir::stablehlo::DotAlgorithmAttr dotAlgorithmAttr = cast<mlir::stablehlo::DotAlgorithmAttr>(attr);
        ss << "(DotAlgorithmAttr ";
        ss << eggifyType(dotAlgorithmAttr.getLhsPrecisionType()) << " ";
        ss << eggifyType(dotAlgorithmAttr.getRhsPrecisionType()) << " ";
        ss << eggifyType(dotAlgorithmAttr.getAccumulationType()) << " ";
        ss << dotAlgorithmAttr.getLhsComponentCount() << " ";
        ss << dotAlgorithmAttr.getRhsComponentCount() << " ";
        ss << dotAlgorithmAttr.getNumPrimitiveOperations() << " ";
        ss << (dotAlgorithmAttr.getAllowImpreciseAccumulation() ? "true" : "false") << ")";
    } else if (typeId == mlir::TypeID::get<mlir::stablehlo::DotDimensionNumbersAttr>()) {
        mlir::stablehlo::DotDimensionNumbersAttr dotDimAttr = cast<mlir::stablehlo::DotDimensionNumbersAttr>(attr);
        ss << "(DotDimensionNumbersAttr (vec-of";
        for (int64_t dim: dotDimAttr.getLhsBatchingDimensions()) {
            ss << " " << dim;
        }
        ss << ") (vec-of";
        for (int64_t dim: dotDimAttr.getRhsBatchingDimensions()) {
            ss << " " << dim;
        }
        ss << ") (vec-of";
        for (int64_t dim: dotDimAttr.getLhsContractingDimensions()) {
            ss << " " << dim;
        }
        ss << ") (vec-of";
        for (int64_t dim: dotDimAttr.getRhsContractingDimensions()) {
            ss << " " << dim;
        }
        ss << "))";

    } else if (egglogCustom.attrStringifiers.find(typeName) != egglogCustom.attrStringifiers.end()) {  // custom attr by user
        AttrStringifyFunction stringifyFunc = egglogCustom.attrStringifiers.at(typeName);
        std::vector<std::string> split = stringifyFunc(attr, *this);
        assert(split.size() > 0);

        ss << "(" << split[0];
        for (size_t i = 1; i < split.size(); i++) {
            ss << " " << split[i];
        }
        ss << ")";

    } else {  // (OpaqueAttr "<attr>" "<type-name>")
        ss << "(OpaqueAttr \"";
        attr.print(ss);
        ss << "\" \"" << typeName << "\")";
    }

    // TODO add DenseResourceElementsAttr
    // TODO add DenseStringElementsAttr
    // TODO add DictionaryAttr
    // TOOD add IntegerSetAttr
    // TODO add OpaqueAttr
    // TODO add SparseElementsAttr
    // TODO add SymbolRefAttr
    // TODO add StridedLayoutAttr

    ss.flush();

    // replace ".000000e+00" with "."
    // std::string::size_type n = 0;
    // while ((n = egglogCode.find(".000000e+00", n)) != std::string::npos) {
    //     egglogCode.replace(n, 11, ".0");
    // }

    return egglogCode;
}

mlir::Value Egglog::parseValue(const std::string& valueStr) {
    std::vector<std::string> split = splitExpression(valueStr);
    assert(split[0] == "Value");

    size_t id = std::stoll(split[1]);
    EggifiedOp* eggifiedOp = findEggifiedOp(id);
    if (eggifiedOp == nullptr) {
        llvm::outs() << "Value " << id << " not found\n";
        exit(1);
    }

    return eggifiedOp->mlirValues[0];
}

mlir::Operation* Egglog::parseOperation(const std::string& newOpStr, mlir::OpBuilder& builder) {
    // llvm::outs() << "PARSING OPERATION: " << newOpStr << "\n";
    std::vector<std::string> split = splitExpression(newOpStr);
    std::string opName = split[0];

    if (opName == "Value") {  // If it's am opaque op, no operation to parse
        return nullptr;
    }

    bool cacheable = opName.find("func_call") == std::string::npos;  // TODO hacky way to check if cacheable, fix this
    if (parsedOps.find(newOpStr) != parsedOps.end() && cacheable) {  // look in the parsed ops cache
        return parsedOps[newOpStr];
    }

    if (supportedEgglogOps.find(opName) == supportedEgglogOps.end()) {  // unsupported operation
        llvm::outs() << "Unsupported operation '" << opName << "'.\n";
        exit(1);
    }

    opName[opName.find_first_of('_')] = '.';  // Replace first underscore with dot
    EgglogOpDef egglogOpDef = supportedEgglogOps.at(opName);
    std::string mlirOpName = egglogOpDef.dialect + "." + egglogOpDef.name;

    size_t index = 0;

    // Operands
    std::vector<mlir::Value> operands;
    for (size_t i = 0; i < egglogOpDef.nOperands; i++, index++) {
        std::string operandStr = split[index + 1];  // get the operand string

        if (operandStr.find("(Value ") == 0) {
            mlir::Value operand = parseValue(operandStr);
            operands.push_back(operand);
        } else {  // new operation
            mlir::Operation* nestedOperand = parseOperation(operandStr, builder);
            mlir::Value operand = nestedOperand->getResult(0);  // TODO support multiple results?
            operands.push_back(operand);
        }
    }

    // attr
    std::vector<mlir::NamedAttribute> attributes;
    for (size_t i = 0; i < egglogOpDef.nAttributes; i++, index++) {
        mlir::NamedAttribute attr = parseNamedAttribute(split[index + 1]);
        attributes.push_back(attr);
    }

    // <region1> <region2> ... <regionR>
    std::vector<std::vector<mlir::Block*>> regions;
    for (size_t i = 0; i < egglogOpDef.nRegions; i++, index++) {
        std::vector<mlir::Block*> blocks = parseBlocksFromRegion(split[index + 1], builder);
        regions.push_back(blocks);
    }

    // Return type
    std::vector<mlir::Type> types;
    for (size_t i = 0; i < egglogOpDef.nResults; i++, index++) {
        mlir::Type type = parseType(split[index + 1]);
        types.push_back(type);
    }

    // Create the operation
    mlir::Operation* newOp = nullptr;

    if (mlirOpName.find("linalg.") == 0) {  // custom ops
        std::string op = mlirOpName.substr(7);
        if (op == "transpose") {
            mlir::Attribute attr = attributes[0].getValue();
            newOp = builder.create<mlir::linalg::TransposeOp>(mlir::UnknownLoc::get(&context), operands[0], operands[1], cast<mlir::DenseI64ArrayAttr>(attr));
        } else if (op == "matmul") {
            newOp = builder.create<mlir::linalg::MatmulOp>(mlir::UnknownLoc::get(&context), operands[2].getType(), llvm::ArrayRef<mlir::Value> {operands[0], operands[1]}, operands[2]);
        }
    } else {  // other ops that have no hidden region, thus are easy to create with OperationState
        mlir::OperationState state(mlir::UnknownLoc::get(&context), mlirOpName);
        state.addOperands(operands);
        state.addAttributes(attributes);
        state.addTypes(types);

        for (size_t i = 0; i < regions.size(); i++) {
            mlir::Region* region = state.addRegion();
            for (mlir::Block* block: regions[i]) {
                region->push_back(block);
            }
        }

        newOp = builder.create(state);
    }

    parsedOps[newOpStr] = newOp;  // cache the parsed operation
    return newOp;
}

std::vector<mlir::Block*> Egglog::parseBlocksFromRegion(const std::string& regionStr, mlir::OpBuilder& builder) {
    std::vector<std::string> split = splitExpression(regionStr);
    assert(split[0] == "Reg");

    std::string blockVecStr = split[1];
    std::vector<std::string> blockStrs = splitExpression(blockVecStr);

    std::vector<mlir::Block*> blocks;
    for (size_t i = 1; i < blockStrs.size(); i++) {
        mlir::Block* block = parseBlock(blockStrs[i], builder);
        blocks.push_back(block);
    }

    return blocks;
}

mlir::Block* Egglog::parseBlock(const std::string& blockStr, mlir::OpBuilder& builder) {
    std::vector<std::string> split = splitExpression(blockStr);
    assert(split[0] == "Blk");

    std::string opVecStr = split[1];
    std::vector<std::string> opStrs = splitExpression(opVecStr);

    mlir::Block* block = new mlir::Block();
    for (size_t i = 1; i < opStrs.size(); i++) {
        mlir::Operation* op = parseOperation(opStrs[i], builder);
        if (op->getBlock()) {
            op->remove();
        }
        block->push_back(op);
    }

    return block;
}

// (Value <id> <type>)
EggifiedOp* Egglog::eggifyValue(mlir::Value value) {
    EggifiedOp* foundEggifiedOp = findEggifiedOp(value);
    if (foundEggifiedOp != nullptr) {
        return foundEggifiedOp;
    }

    std::string type = eggifyType(value.getType());
    size_t id = nextId();
    std::string egglogOp = "(Value " + std::to_string(id) + " " + type + ")";

    EggifiedOp* eggifiedValue = EggifiedOp::value(id, egglogOp, value);
    eggifiedBlock.push_back(eggifiedValue);
    return eggifiedValue;
}

EggifiedOp* Egglog::eggifyOpaqueOperation(mlir::Operation* op) {
    if (op->getNumResults() == 1) {
        return eggifyValue(op->getResult(0));  // backwards compatibility
    }

    size_t id = nextId();
    std::string egglogOp = "(Value " + std::to_string(id) + " (None))";

    EggifiedOp* eggifiedValue = EggifiedOp::opaqueOp(id, egglogOp, op);
    eggifiedBlock.push_back(eggifiedValue);
    return eggifiedValue;
}

// (<op> <operand1> <operand2> ... <operandN> <attr1> <attr2> ... <attrM> <region1> <region2> ... <regionR> <type1> <type2> ... <typeT>)
EggifiedOp* Egglog::eggifyOperation(mlir::Operation* op) {

    EggifiedOp* foundEggifiedOp = findEggifiedOp(op);
    if (foundEggifiedOp != nullptr) {
        return foundEggifiedOp;
    }

    std::string opName = op->getName().getStringRef().str();
    bool isSupported = supportedEgglogOps.find(opName) != supportedEgglogOps.end();

    if (!isSupported) {
        // check is supported if we add the number of operands
        // TODO change to lookup in the map for the appropriate operation name
        std::string opNameWithNumOperands = opName + "." + std::to_string(op->getNumOperands());
        isSupported = supportedEgglogOps.find(opNameWithNumOperands) != supportedEgglogOps.end();

        if (!isSupported) {
            LLVM_DEBUG(llvm::dbgs() << "Unsupported operation '" << opName << "' and '" << opNameWithNumOperands << "' but using the result as a variable.\n");
            return eggifyOpaqueOperation(op);
        }

        opName = opNameWithNumOperands;
    }

    EgglogOpDef egglogOpDef = supportedEgglogOps.at(opName);

    // check if not the same number of operands
    if (egglogOpDef.nOperands != op->getNumOperands()) {
        llvm::outs() << "Unsupported operation '" << opName << "' since it has " << op->getNumOperands() << " operands but egglog's '" << egglogOpDef.fullName << "' expects " << egglogOpDef.nOperands << " operands.\n";
        return eggifyOpaqueOperation(op);
    }

    // check if not the same number of results
    if (egglogOpDef.nResults != op->getNumResults()) {
        llvm::outs() << "Unsupported operation '" << opName << "' since it has " << op->getNumResults() << " results but egglog's '" << egglogOpDef.fullName << "' expects " << egglogOpDef.nResults << " results.\n";
        return eggifyOpaqueOperation(op);
    }

    op->removeAttr("linalg.memoized_indexing_maps");  // temporary fix for linalg ops

    std::stringstream ss;

    std::string cleanOpName = opName;
    std::replace(cleanOpName.begin(), cleanOpName.end(), '.', '_');

    ss << "(" << cleanOpName;  // (<op>

    // <operand1> <operand2> ... <operandN>
    std::vector<EggifiedOp*> operands;
    for (size_t i = 0; i < egglogOpDef.nOperands; i++) {
        EggifiedOp* eggifiedOperand = eggifyValue(op->getOperand(i));
        operands.push_back(eggifiedOperand);
        ss << " " << eggifiedOperand->getPrintId();
    }

    // <attr1> <attr2> ... <attrM>
    llvm::ArrayRef<mlir::NamedAttribute> attrs = op->getAttrs();
    for (size_t i = 0; i < egglogOpDef.nAttributes; i++) {
        ss << " " << eggifyNamedAttribute(attrs[i]);
    }

    // <region1> <region2> ... <regionR>
    for (size_t i = 0; i < egglogOpDef.nRegions; i++) {
        ss << " " << eggifyRegion(op->getRegion(i));
    }

    // <type1> <type2> ... <typeT>
    for (size_t i = 0; i < egglogOpDef.nResults; i++) {
        ss << " " << eggifyType(op->getResult(i).getType());
    }

    ss << ")";

    EggifiedOp* eggifiedOp = EggifiedOp::op(nextId(), ss.str(), operands, op);
    for (EggifiedOp* operand: eggifiedOp->operands) {
        operand->addUser(eggifiedOp);
    }

    eggifiedBlock.push_back(eggifiedOp);
    return eggifiedOp;
}

std::string Egglog::eggifyBlock(mlir::Block& block) {
    std::stringstream ss;

    mlir::Block::OpListType& ops = block.getOperations();

    ss << "(Blk (vec-of";
    for (mlir::Operation& op: ops) {
        if (op.getNumResults() > 1) {  // exit if more than one result
            llvm::outs() << "Skipping operation with more than one result: " << op.getName() << "\n";
        }

        ss << " " << eggifyOperation(&op)->getPrintId();
    }
    ss << "))";
    return ss.str();
}

std::string Egglog::eggifyRegion(mlir::Region& region) {
    std::stringstream ss;

    mlir::Region::BlockListType& blocks = region.getBlocks();

    ss << "(Reg (vec-of";
    for (mlir::Block& block: blocks) {
        ss << " " << eggifyBlock(block);
    }
    ss << "))";
    return ss.str();
}

EggifiedOp* Egglog::findEggifiedOp(mlir::Operation* op) {
    for (EggifiedOp* eggifiedOp: eggifiedBlock) {
        if (eggifiedOp->mlirOp == op) {
            return eggifiedOp;
        }
    }
    return nullptr;
}
EggifiedOp* Egglog::findEggifiedOp(mlir::Value value) {
    for (EggifiedOp* eggifiedOp: eggifiedBlock) {
        std::vector<mlir::Value>& mlirValues = eggifiedOp->mlirValues;
        if (std::find(mlirValues.begin(), mlirValues.end(), value) != mlirValues.end()) {
            return eggifiedOp;
        }
    }
    return nullptr;
}
EggifiedOp* Egglog::findEggifiedOp(size_t id) {
    for (EggifiedOp* eggifiedOp: eggifiedBlock) {
        if (eggifiedOp->id == id) {
            return eggifiedOp;
        }
    }
    return nullptr;
}
EggifiedOp* Egglog::findEggifiedOp(const std::string& egglogOp) {
    for (EggifiedOp* eggifiedOp: eggifiedBlock) {
        if (eggifiedOp->egglogOp == egglogOp) {
            return eggifiedOp;
        }
    }
    return nullptr;
}
