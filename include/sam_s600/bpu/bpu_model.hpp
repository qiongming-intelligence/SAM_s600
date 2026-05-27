#pragma once

#include <string>
#include <vector>

#include "sam_s600/core/tensor.hpp"

namespace sam_s600 {

class BpuModel {
 public:
  BpuModel() = default;
  explicit BpuModel(std::string model_path);

  void Load(std::string model_path);

  [[nodiscard]] bool Loaded() const;
  [[nodiscard]] const std::string& Path() const;
  [[nodiscard]] const std::string& Name() const;
  [[nodiscard]] const std::vector<TensorInfo>& Inputs() const;
  [[nodiscard]] const std::vector<TensorInfo>& Outputs() const;

 private:
  std::string path_;
  std::string name_;
  std::vector<TensorInfo> inputs_;
  std::vector<TensorInfo> outputs_;
  bool loaded_{false};
};

}  // namespace sam_s600
