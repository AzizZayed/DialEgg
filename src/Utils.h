#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

inline long long svtoll(std::string_view str) {
    return std::stoll(std::string(str));
}

inline int svtoi(std::string_view str) {
    return std::stoi(std::string(str));
}

inline double svtod(std::string_view str) {
    return std::stod(std::string(str));
}

inline float svtof(std::string_view str) {
    return std::stof(std::string(str));
}

inline bool svtob(std::string_view str) {
    if (str == "true" || str == "1") {
        return true;
    } else if (str == "false" || str == "0") {
        return false;
    } else {
        llvm::outs() << "Invalid boolean string: " << str << "\n";
        exit(1);
    }
}

inline std::string doubleToString(double d) {
    std::ostringstream ss;
    ss << std::fixed << d;
    std::string str = ss.str();

    // remove trailing zeros
    size_t lastNonZero = str.find_last_not_of('0');
    if (lastNonZero != std::string::npos && str[lastNonZero] == '.') {
        lastNonZero++;
    }
    str = str.substr(0, lastNonZero + 1);
    return str;
}

/** Returns the given string without the wrapping character */
inline std::string_view unwrap(std::string_view str, char c = ' ') {
    if (str.front() == c && str.back() == c) {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

/** Returns the given string without being wrapped in brackets */
inline std::string_view unwrapBrackets(std::string_view str) {
    if (str.front() == '(' && str.back() == ')') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

inline bool isBlank(const std::string& str) {
    return str.find_first_not_of(' ') == std::string::npos;
}

inline bool isBlank(const std::string_view& str) {
    return str.find_first_not_of(' ') == std::string_view::npos;
}

inline void printFileContents(const std::string& filename, llvm::raw_ostream& ros = llvm::outs()) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            ros << line << "\n";
        }
        file.close();
    }
}

#endif  //UTILS_H