#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sam_s600/core/tensor.hpp"

namespace sam_s600 {

struct BpuTensor {
  TensorInfo info;
  std::vector<std::uint8_t> host_data;

  [[nodiscard]] std::size_t Size() const { return host_data.size(); }
  [[nodiscard]] bool Empty() const { return host_data.empty(); }
};

}  // namespace sam_s600
