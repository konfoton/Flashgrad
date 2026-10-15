#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flashgrad {

// Sizes and strides are counted in elements, not bytes (like PyTorch).
using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

// Product of the sizes; 1 for a 0-dimensional shape. Throws on negative sizes or overflow.
std::int64_t numel(const Shape& shape);

// Row-major (C-order) strides for `shape`.
Strides contiguous_strides(const Shape& shape);

// NumPy-style broadcasting of two shapes (aligned from the right; size-1 dims stretch).
// `op` names the calling operation in the error message.
Shape broadcast_shapes(const Shape& a, const Shape& b, const char* op = "broadcast_shapes");

// Wraps a possibly negative dimension index into [0, ndim). Throws when out of range.
std::int64_t normalize_dim(std::int64_t dim, std::int64_t ndim, const char* op);

// "[2, 3]".
std::string to_string(const Shape& shape);

}  // namespace flashgrad
