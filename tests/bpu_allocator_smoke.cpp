#include "sam_s600/bpu/bpu_allocator.hpp"

#include <cstdint>
#include <stdexcept>

int main() {
  sam_s600::BpuAllocator allocator;
  sam_s600::BpuAllocationOptions options;
  options.location = sam_s600::BpuMemoryLocation::kHost;
  options.label = "allocator_smoke";

  auto buffer = allocator.Allocate(4096, options);
  if (buffer.Empty() || buffer.Size() != 4096 || buffer.CpuData() == nullptr || buffer.BpuBacked()) {
    throw std::runtime_error("host allocation state is invalid");
  }

  buffer.CpuData()[0] = 0x12;
  buffer.CpuData()[4095] = 0x34;
  if (buffer.CpuData()[0] != 0x12 || buffer.CpuData()[4095] != 0x34) {
    throw std::runtime_error("host allocation is not writable");
  }

  auto moved = std::move(buffer);
  if (moved.Empty() || moved.Size() != 4096 || moved.CpuData() == nullptr) {
    throw std::runtime_error("moved allocation state is invalid");
  }

  moved.Reset();
  if (!moved.Empty() || moved.CpuData() != nullptr) {
    throw std::runtime_error("reset allocation state is invalid");
  }

  return 0;
}
