#ifndef EGGLOG_H
#define EGGLOG_H

#include <map>
#include <set>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

class Egglog;

using AttrStringifyFunction = std::function<std::vector<std::string>(mlir::Attribute, Egglog&)>;
using AttrParseFunction = std::function<mlir::Attribute(const std::vector<std::string_view>&, Egglog&)>;

using TypeStringifyFunction = std::function<std::vector<std::string>(mlir::Type, Egglog&)>;
using TypeParseFunction = std::function<mlir::Type(const std::vector<std::string_view>&, Egglog&)>;

/**
 * Holds the functions for stringifying and parsing custom attributes and types.
 * To register stringifier: `stringifier[T::name.str()] = stringify<T>;`
 * To register parser: `parser["<EggT>"] = parse<T>;`
 */
struct EgglogCustomDefs {
    std::map<std::string, AttrStringifyFunction, std::less<>> attrStringifiers;
    std::map<std::string, AttrParseFunction, std::less<>> attrParsers;

    std::map<std::string, TypeStringifyFunction, std::less<>> typeStringifiers;
    std::map<std::string, TypeParseFunction, std::less<>> typeParsers;
};

/** Holds the information about a custom egglog operation */
struct EgglogOpDef {
    std::string str;
    std::string dialect, name, version;
    std::vector<std::string> args;

    size_t nOperands;
    size_t nAttributes;
    size_t nRegions;
    size_t nResults;

    size_t cost = 1;

    std::string egglogName() const {
        return dialect + "_" + name + (version.empty() ? "" : "_" + version);
    }

    std::string mlirName() const {
        return dialect + "." + name;
    }

    bool matches(mlir::Operation* op) const {
        if (op->getName().getStringRef() != mlirName()) {
            return false;
        }
        if (op->getNumOperands() != nOperands || op->getNumResults() != nResults) {
            return false;
        }
        return true;
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

    // true if this is a `(Get <op> <index>)` projection of one result of a multi-result operation
    bool projection = false;

    // true if the operation is inside a region of another eggified operation: it is rebuilt as part of that operation, never on its own
    bool nested = false;

    // true if an unsupported (opaque) operation uses a result: such users are not eggified, so this is read from the MLIR use list
    bool hasUnsupportedUser = false;

    EggifiedOp(size_t id, bool opaque, std::string egglogOp, const std::vector<EggifiedOp*>& operands, const std::vector<mlir::Value>& mlirValues, mlir::Operation* mlirOp)
        : id(id), opaque(opaque), egglogOp(egglogOp), operands(operands), mlirValues(mlirValues), mlirOp(mlirOp) {}

    static EggifiedOp* opaqueOp(size_t id, std::string egglogOp, mlir::Operation* mlirOp) {
        return new EggifiedOp(id, true, egglogOp, {}, std::vector<mlir::Value>(mlirOp->result_begin(), mlirOp->result_end()), mlirOp);
    }
    static EggifiedOp* value(size_t id, std::string egglogOp, mlir::Value mlirValue) {
        return new EggifiedOp(id, true, egglogOp, {}, {mlirValue}, mlirValue.getDefiningOp());
    }
    /** A `(Get <op> <index>)` term standing for result `index` (`mlirValue`) of the multi-result operation `owner`. */
    static EggifiedOp* resultProjection(size_t id, std::string egglogOp, EggifiedOp* owner, mlir::Value mlirValue) {
        EggifiedOp* eggified = new EggifiedOp(id, false, egglogOp, {owner}, {mlirValue}, nullptr);
        eggified->projection = true;
        return eggified;
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
        return extractionKind() != ExtractionKind::None;
    }

    /** Roots are rebuilt on their own. Nested operations with an unsupported (opaque) user only provide the replacement of their result for that user. */
    enum class ExtractionKind { None,
                                Root,
                                ForOpaqueUsers };
    ExtractionKind extractionKind() const {
        if (opaque || projection) {
            return ExtractionKind::None;
        }
        if (nested) {
            return hasUnsupportedUser ? ExtractionKind::ForOpaqueUsers : ExtractionKind::None;
        }
        return hasNonOpaqueUser() ? ExtractionKind::None : ExtractionKind::Root;
    }

    /** Whether a user that is not opaque consumes this operation, looking through result projections. */
    bool hasNonOpaqueUser() const {
        for (const EggifiedOp* user: users) {
            if (user->projection) {
                if (user->hasNonOpaqueUser()) {
                    return true;
                }
            } else if (!user->opaque) {
                return true;
            }
        }
        return false;
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
        } else if (mlirValues.empty()) {
            os << "<block argument declaration>";
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
    static std::vector<std::string_view> splitExpression(std::string_view opStr);
    
    static std::string removeComment(const std::string& str);

    Egglog(mlir::MLIRContext& context, const EgglogCustomDefs& egglogCustom, const std::map<std::string, EgglogOpDef, std::less<>>& supportedEgglogOps)
        : context(context), egglogCustom(egglogCustom), supportedEgglogOps(supportedEgglogOps) {}

    ~Egglog() {
        for (EggifiedOp* op: eggifiedBlock) {
            delete op;
        }
    }

    size_t nextId() {
        return opId++;
    }

    bool isSupportedOp(std::string_view opName) {
        return supportedEgglogOps.find(opName) != supportedEgglogOps.end();
    }

    /** Parses the given type string into an MLIR type Form (F16) or (Complex <type>) */
    mlir::Type parseType(std::string_view);
    std::string eggifyType(mlir::Type);

    /** Parses the given attribute string into an MLIR named attribute. Form (NamedAttr "<name>" <attr>) */
    mlir::NamedAttribute parseNamedAttribute(std::string_view);
    std::string eggifyNamedAttribute(mlir::NamedAttribute);

    /** Parses the given attribute string into an MLIR attribute. Form (<type> <arg1> <arg2> ... <argN>) */
    mlir::Attribute parseAttribute(std::string_view);
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
    mlir::Operation* parseOperation(std::string_view, mlir::OpBuilder&);
    mlir::Value parseValue(std::string_view);
    /** Parses an operand: `(Value <id> <type>)`, `(Get <op> <index>)` (one result of a multi-result op) or a nested operation (its first result). */
    mlir::Value parseOperand(std::string_view, mlir::OpBuilder&);
    mlir::Block* parseBlock(std::string_view, mlir::OpBuilder&);
    std::vector<mlir::Block*> parseBlocksFromRegion(std::string_view, mlir::OpBuilder&);

    template<typename T, size_t N>
    llvm::SmallVector<T, N> parseVector(std::string_view, std::function<T(std::string_view)>);

    EggifiedOp* eggifyValue(mlir::Value);
    EggifiedOp* eggifyOperation(mlir::Operation*);
    EggifiedOp* eggifyOpaqueOperation(mlir::Operation*);
    std::string eggifyBlock(mlir::Block&);
    std::string eggifyRegion(mlir::Region&);

    template<typename T, typename U>
    void eggifyIterable(llvm::raw_string_ostream& ss, U range);

    /** Sets `hasUnsupportedUser` on all eggified operations; call once everything is eggified. */
    void markUnsupportedUsers();

    std::optional<EgglogOpDef> findEgglogOpDef(mlir::Operation* op);
    EggifiedOp* findEggifiedOp(mlir::Operation*);
    EggifiedOp* findEggifiedOp(mlir::Value);
    EggifiedOp* findEggifiedOp(size_t);
    EggifiedOp* findEggifiedOp(const std::string&);

public:
    size_t opId = 0;
    size_t regionDepth = 0;  // number of regions being eggified around the current operation
    mlir::MLIRContext& context;
    const EgglogCustomDefs& egglogCustom;
    const std::map<std::string, EgglogOpDef, std::less<>>& supportedEgglogOps;  // Supported operations

    // caches
    std::vector<EggifiedOp*> eggifiedBlock;
    std::map<std::string_view, mlir::Operation*> parsedOps;

    /** Operations parsed while rebuilding one block: a block must not share operations with its siblings (e.g. both branches of an scf.if). */
    struct ParseScope {
        std::map<std::string_view, mlir::Operation*> ops;
        std::vector<mlir::Operation*> order;  // creation order
    };
    std::vector<ParseScope> parseScopes;

    /** Unsupported (opaque) operations found inside regions: they are not rebuilt but moved into the rebuilt block. */
    std::set<mlir::Operation*> regionOpaqueOps;
    /** Block arguments of rebuilt blocks: original argument -> argument of the rebuilt block. */
    llvm::DenseMap<mlir::Value, mlir::Value> argRemap;
    /** Rebuilt equivalent (egglog term) of operations inside regions that have unsupported users, which must be repointed to it. */
    std::map<mlir::Operation*, std::string> opaqueUserReplacements;
};

#endif  //EGGLOG_H