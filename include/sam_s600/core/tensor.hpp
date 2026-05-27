#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam_s600 {

enum class TensorDataType {
  kUnknown,
  kUint8,
  kInt32,
  kFloat32,
};

struct TensorShape {
  std::vector<int> dims;
};

struct TensorInfo {
  std::string name;
  TensorShape shape;
  TensorDataType dtype{TensorDataType::kUnknown};
  std::uint64_t byte_size{0};
};

}  // namespace sam_s600
