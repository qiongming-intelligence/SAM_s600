#include "sam_s600/bpu/bpu_allocator.hpp"

namespace sam_s600 {

BpuBuffer BpuAllocator::Allocate(std::size_t bytes) const {
  BpuBuffer buffer;
  buffer.data = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[bytes]{});
  buffer.size = bytes;
  return buffer;
}

}  // namespace sam_s600
