#ifndef EGGLOG_H
#define EGGLOG_H

#include <map>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

class Egglog;

using AttrStringifyFunction = std::function<std::vector<std::string>(mlir::Attribute, Egglog&)>;
using AttrParseFunction = std::function<mlir::Attribute(const std::vector<std::string>&, Egglog&)>;

using TypeStringifyFunction = std::function<std::vector<std::string>(mlir::Type, Egglog&)>;
using TypeParseFunction = std::function<mlir::Type(const std::vector<std::string>&, Egglog&)>;

/**
 * Holds the functions for stringifying and parsing custom attributes and types.
 * To register stringifier: `stringifier[T::name.str()] = stringify<T>;`
 * To register parser: `parser["<EggT>"] = parse<T>;`
 */
struct EgglogCustomDefs {
    std::map<std::string, AttrStringifyFunction> attrStringifiers;
    std::map<std::string, AttrParseFunction> attrParsers;

    std::map<std::string, TypeStringifyFunction> typeStringifiers;
    std::map<std::string, TypeParseFunction> typeParsers;
};

/** Holds the information about a custom egglog operation */
struct EgglogOpDef {
    std::string str;
    std::string fullName, dialect, name, version;
    std::vector<std::string> args;

    size_t nOperands;
    size_t nAttributes;
    size_t nRegions;
    size_t nResults;

    size_t cost = 1;

    std::string egglogName() const {
        return dialect + "_" + name;
    }

    std::string mlirName() const {
        return dialect + "." + name;
    }

    // format: (function [name] ([params]) Op) or (function [name] ([params]) Op :cost [cost])
    static bool isOpFunction(const std::string& opStr);
    static EgglogOpDef parseOpFunction(const std::string& opStr);
};

/** An MLIR op or value that has been eggified */
struct EggifiedOp {
    size_t id;
    bool opaque;
    std::string egglogOp;
    std::vector<EggifiedOp*> operands;
    std::vector<EggifiedOp*> users;

    // for reference
    std::vector<mlir::Value> mlirValues;
    mlir::Operation* mlirOp;

    EggifiedOp(size_t id, bool opaque, std::string egglogOp, const std::vector<EggifiedOp*>& operands, const std::vector<mlir::Value>& mlirValues, mlir::Operation* mlirOp)
        : id(id), opaque(opaque), egglogOp(egglogOp), operands(operands), mlirValues(mlirValues), mlirOp(mlirOp) {}

    static EggifiedOp* opaqueOp(size_t id, std::string egglogOp, mlir::Operation* mlirOp) {
        return new EggifiedOp(id, true, egglogOp, {}, std::vector<mlir::Value>(mlirOp->result_begin(), mlirOp->result_end()), mlirOp);
    }
    static EggifiedOp* value(size_t id, std::string egglogOp, mlir::Value mlirValue) {
        return new EggifiedOp(id, true, egglogOp, {}, {mlirValue}, mlirValue.getDefiningOp());
    }
    static EggifiedOp* op(size_t id, std::string egglogOp, const std::vector<EggifiedOp*>& operands, mlir::Operation* mlirOp) {
        return new EggifiedOp(id, false, egglogOp, operands, std::vector<mlir::Value>(mlirOp->result_begin(), mlirOp->result_end()), mlirOp);
    }

    /**
     * We want to avoid extraction as much as possible because Egglog's extraction algorithm is slow.
     * 
     * Opaque ops are never touched and thus do not need to be extracted.
     * Operations with no users must be extracted, in case they were optimized by Egglog.
     * Operations with all users as opaque must be extracted, in case they were optimized by Egglog.
     * Operations with a non-opaque user may not be extracted, as the optimized operation will be inlined in the user.
     */
    bool shouldBeExtracted() const {
        if (opaque) {  // do no extract opaque operations
            return false;
        }

        // extract operation that have all users as opaque (in other words, do not extract if there is a user that is not opaque)
        for (const EggifiedOp* user: users) {
            if (!user->opaque) {
                return false;
            }
        }

        return true;
    }

    void addUser(EggifiedOp* user) {
        users.push_back(user);
    }

    std::string getPrintId() const {
        return "op" + std::to_string(id);
    }

    std::string egglogLet() const {
        return "(let " + getPrintId() + " " + egglogOp + ")";
    }

    void print(llvm::raw_ostream& os) const {
        os << "[" << id << "] " << egglogOp << " FROM OP: ";
        mlirDump(os);

        // print number of users
        os << "; USERS: ";
        for (const EggifiedOp* user: users) {
            os << user->getPrintId() << " ";
        }

        os << "\n";
    }

    void mlirDump(llvm::raw_ostream& os) const {
        if (mlirOp != nullptr) {
            os << *mlirOp;
        } else {
            os << mlirValues[0];  // todo fix for multi-result support
        }
    }
};

/**
 * Holds all egglog informations and operations.
 * The Egglog expressions are in the form of S-expressions.
 * Example: `(arith_constant (NamedAttr "value" (FloatAttr 1.0 "f64")) "f64")` -> `arith.constant 1.0 : f64`
 */
class Egglog {
public:
    /**
     * Splits the given egglog expression string into a list of strings: function name and parameters.
     * Example: "(linalg_add (linalg_abs (tensor_empty)) (tensor_empty))" -> ["linalg_add", "(linalg_abs (tensor_empty))", "(tensor_empty)"]
     */
    static std::vector<std::string> splitExpression(std::string opStr);
    
    static std::string removeComment(const std::string& str);

    Egglog(mlir::MLIRContext& context, const EgglogCustomDefs& egglogCustom, const std::map<std::string, EgglogOpDef>& supportedEgglogOps)
        : context(context), egglogCustom(egglogCustom), supportedEgglogOps(supportedEgglogOps) {}

    ~Egglog() {
        for (EggifiedOp* op: eggifiedBlock) {
            delete op;
        }
    }

    size_t nextId() {
        return opId++;
    }

    bool isSupportedOp(const std::string& opName) {
        return supportedEgglogOps.find(opName) != supportedEgglogOps.end();
    }

    /** Parses the given type string into an MLIR type Form (F16) or (Complex <type>) */
    mlir::Type parseType(std::string);
    std::string eggifyType(mlir::Type);

    /** Parses the given attribute string into an MLIR named attribute. Form (NamedAttr "<name>" <attr>) */
    mlir::NamedAttribute parseNamedAttribute(const std::string&);
    std::string eggifyNamedAttribute(mlir::NamedAttribute);

    /** Parses the given attribute string into an MLIR attribute. Form (<type> <arg1> <arg2> ... <argN>) */
    mlir::Attribute parseAttribute(const std::string&);
    std::string eggifyAttribute(mlir::Attribute);

    /**
     * Parses the given operation string into a list of MLIR operations, in order or dependency.
     * Example:
     *      (linalg_transpose (tensor_empty "tensor<3x2xf32>") (tensor_empty "tensor<2x3xf32>") "tensor<2x3xf32>")
     * Becomes:
     *      %0 = tensor.empty() : tensor<3x2xf32>
     *      %1 = tensor.empty() : tensor<2x3xf32>
     *      %2 = linalg.transpose %0, %1 : tensor<3x2xf32>, tensor<2x3xf32>
     */
    mlir::Operation* parseOperation(const std::string&, mlir::OpBuilder&);
    mlir::Value parseValue(const std::string&);
    mlir::Block* parseBlock(const std::string&, mlir::OpBuilder&);
    std::vector<mlir::Block*> parseBlocksFromRegion(const std::string&, mlir::OpBuilder&);

    EggifiedOp* eggifyValue(mlir::Value);
    EggifiedOp* eggifyOperation(mlir::Operation*);
    EggifiedOp* eggifyOpaqueOperation(mlir::Operation*);
    std::string eggifyBlock(mlir::Block&);
    std::string eggifyRegion(mlir::Region&);

    EggifiedOp* findEggifiedOp(mlir::Operation*);
    EggifiedOp* findEggifiedOp(mlir::Value);
    EggifiedOp* findEggifiedOp(size_t);
    EggifiedOp* findEggifiedOp(const std::string&);

public:
    size_t opId = 0;
    mlir::MLIRContext& context;
    const EgglogCustomDefs& egglogCustom;
    const std::map<std::string, EgglogOpDef>& supportedEgglogOps;  // Supported operations

    // caches
    std::vector<EggifiedOp*> eggifiedBlock;
    std::map<std::string, mlir::Operation*> parsedOps;
};

#endif  //EGGLOG_H