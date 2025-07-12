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

template<typename T>
inline T convertFromString(const std::string& str);

// specializations
template<>
inline double convertFromString<double>(const std::string& str) {
    return std::stod(str);
}

template<>
inline float convertFromString<float>(const std::string& str) {
    return std::stof(str);
}

template<>
inline int32_t convertFromString<int32_t>(const std::string& str) {
    return std::stoi(str);
}

template<>
inline int64_t convertFromString<int64_t>(const std::string& str) {
    return std::stoll(str);
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
inline std::string unwrap(const std::string& str, char c = ' ') {
    if (str.front() == c && str.back() == c) {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

/** Returns the given string without being wrapped in brackets */
inline std::string unwrapBrackets(const std::string& str) {
    if (str.front() == '(' && str.back() == ')') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

inline bool isBlank(const std::string& str) {
    return str.find_first_not_of(' ') == std::string::npos;
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