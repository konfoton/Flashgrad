#pragma once

#include <cstdint>

// Operation identifiers shared by the core dispatch layer and the backends.
namespace flashgrad {

enum class BinaryOp : std::uint8_t { Add, Sub, Mul, Div, Maximum, Minimum, Pow };
enum class CompareOp : std::uint8_t { Eq, Ne, Lt, Le, Gt, Ge };
enum class UnaryOp : std::uint8_t { Neg, Abs, Relu, Exp, Log, Sqrt, Tanh, Sigmoid };
enum class ReduceOp : std::uint8_t { Sum, Mean, Max, Min };
enum class ArgReduceOp : std::uint8_t { Argmax, Argmin };

}  // namespace flashgrad
