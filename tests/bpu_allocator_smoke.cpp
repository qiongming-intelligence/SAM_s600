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

  sam_s600::TensorInfo tensor;
  tensor.name = "tensor_smoke";
  tensor.shape.dims = {1, 3, 8, 8};
  tensor.dtype = sam_s600::TensorDataType::kFloat32;
  if (sam_s600::TensorStorageBytes(tensor) != 768) {
    throw std::runtime_error("tensor byte inference is invalid");
  }

  auto tensor_buffer = allocator.AllocateTensor(tensor, options);
  if (tensor_buffer.info.name != tensor.name || tensor_buffer.buffer.Size() != 768 || tensor_buffer.buffer.CpuData() == nullptr) {
    throw std::runtime_error("tensor allocation state is invalid");
  }

  tensor.byte_size = 1024;
  if (sam_s600::TensorStorageBytes(tensor) != 1024) {
    throw std::runtime_error("aligned tensor byte size is ignored");
  }

  return 0;
}
