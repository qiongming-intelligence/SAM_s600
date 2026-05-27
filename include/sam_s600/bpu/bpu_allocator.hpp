#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace sam_s600 {

struct BpuBuffer {
  std::shared_ptr<std::uint8_t[]> data;
  std::size_t size{0};
};

class BpuAllocator {
 public:
  [[nodiscard]] BpuBuffer Allocate(std::size_t bytes) const;
};

}  // namespace sam_s600
