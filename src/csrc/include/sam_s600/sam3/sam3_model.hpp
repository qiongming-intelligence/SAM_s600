#pragma once

#include <string>
#include <vector>

#include "sam_s600/bpu/bpu_context.hpp"
#include "sam_s600/bpu/bpu_model.hpp"
#include "sam_s600/core/tensor.hpp"
#include "sam_s600/sam3/sam3_config.hpp"

namespace sam_s600 {

struct Sam3ModelPartRuntimeStatus {
  std::string name;
  std::string path;
  bool exists{false};
  bool loaded{false};
  std::string error;
  std::string model_name;
  int compile_bpu_core_num{0};
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
};

class Sam3Model {
 public:
  explicit Sam3Model(Sam3Config config = {}, BpuRuntimeOptions runtime = {});

  void Load(bool require_all = false);
  void LoadParts(const std::vector<std::string>& names, bool require = true);
  void LoadPart(const std::string& name, bool require = true);
  void Reset();

  [[nodiscard]] const Sam3Config& Config() const;
  [[nodiscard]] bool Loaded() const;
  [[nodiscard]] const std::vector<Sam3ModelPartRuntimeStatus>& PartStatuses() const;
  [[nodiscard]] const BpuModel* FindPart(const std::string& name) const;

 private:
  struct LoadedPart {
    std::string name;
    BpuModel model;
  };

  Sam3Config config_;
  BpuContext context_;
  std::vector<LoadedPart> loaded_parts_;
  std::vector<Sam3ModelPartRuntimeStatus> part_statuses_;
  bool loaded_{false};
};

}  // namespace sam_s600
