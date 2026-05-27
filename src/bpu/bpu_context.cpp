#include "sam_s600/bpu/bpu_context.hpp"

namespace sam_s600 {

BpuContext::BpuContext(BpuRuntimeOptions options) : options_(std::move(options)) {}

const BpuRuntimeOptions& BpuContext::Options() const { return options_; }

bool BpuContext::IsAvailable() const {
#if defined(SAM_S600_HAS_HOBOT)
  return true;
#else
  return false;
#endif
}

std::string BpuContext::RuntimeName() const { return IsAvailable() ? "D-Robotics S600 BPU" : "stub"; }

}  // namespace sam_s600
