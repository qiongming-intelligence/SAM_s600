#pragma once

#include "sam_s600/bpu/bpu_context.hpp"
#include "sam_s600/bpu/bpu_model.hpp"
#include "sam_s600/sam3/sam3_config.hpp"

namespace sam_s600 {

class Sam3Model {
 public:
  explicit Sam3Model(Sam3Config config = {}, BpuRuntimeOptions runtime = {});

  void Load();

  [[nodiscard]] const Sam3Config& Config() const;
  [[nodiscard]] bool Loaded() const;

 private:
  Sam3Config config_;
  BpuContext context_;
  bool loaded_{false};
};

}  // namespace sam_s600
