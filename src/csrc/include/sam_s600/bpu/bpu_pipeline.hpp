#pragma once

#include "sam_s600/bpu/bpu_context.hpp"
#include "sam_s600/bpu/bpu_model.hpp"

namespace sam_s600 {

struct BpuPipelineConfig {
  BpuRuntimeOptions runtime;
};

class BpuPipeline {
 public:
  explicit BpuPipeline(BpuPipelineConfig config = {}) : context_(config.runtime) {}

  [[nodiscard]] const BpuContext& Context() const { return context_; }

 private:
  BpuContext context_;
};

}  // namespace sam_s600
