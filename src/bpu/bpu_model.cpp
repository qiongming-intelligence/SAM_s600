#include "sam_s600/bpu/bpu_model.hpp"

#include <utility>

namespace sam_s600 {

BpuModel::BpuModel(std::string model_path) { Load(std::move(model_path)); }

void BpuModel::Load(std::string model_path) {
  path_ = std::move(model_path);
  const auto slash = path_.find_last_of('/');
  name_ = slash == std::string::npos ? path_ : path_.substr(slash + 1);
  loaded_ = !path_.empty();
}

bool BpuModel::Loaded() const { return loaded_; }

const std::string& BpuModel::Path() const { return path_; }

const std::string& BpuModel::Name() const { return name_; }

const std::vector<TensorInfo>& BpuModel::Inputs() const { return inputs_; }

const std::vector<TensorInfo>& BpuModel::Outputs() const { return outputs_; }

}  // namespace sam_s600
