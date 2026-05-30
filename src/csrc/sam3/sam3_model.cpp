#include "sam_s600/sam3/sam3_model.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "sam_s600/sam3/sam3_manifest.hpp"

namespace sam_s600 {

Sam3Model::Sam3Model(Sam3Config config, BpuRuntimeOptions runtime)
    : config_(std::move(config)), context_(std::move(runtime)) {}

void Sam3Model::Load(bool require_all) {
  Reset();

  for (const auto& part : ListSam3ModelParts(config_)) {
    Sam3ModelPartRuntimeStatus status;
    status.name = part.name;
    status.path = part.path;
    status.exists = std::filesystem::exists(part.path);

    if (!status.exists) {
      status.error = "missing";
      part_statuses_.push_back(std::move(status));
      if (require_all) {
        throw std::runtime_error("missing SAM3 model part: " + part.name + " -> " + part.path);
      }
      continue;
    }

    try {
      BpuModel model(part.path);
      status.loaded = true;
      status.model_name = model.Name();
      status.compile_bpu_core_num = model.CompileBpuCoreNum();
      status.inputs = model.Inputs();
      status.outputs = model.Outputs();
      loaded_parts_.push_back(LoadedPart{part.name, std::move(model)});
    } catch (const std::exception& error) {
      status.error = error.what();
      if (require_all) {
        throw;
      }
    }

    part_statuses_.push_back(std::move(status));
  }

  loaded_ = !loaded_parts_.empty();
}

void Sam3Model::LoadParts(const std::vector<std::string>& names, bool require) {
  for (const auto& name : names) {
    LoadPart(name, require);
  }
}

void Sam3Model::LoadPart(const std::string& name, bool require) {
  for (const auto& loaded : loaded_parts_) {
    if (loaded.name == name) {
      return;
    }
  }

  for (const auto& part : ListSam3ModelParts(config_)) {
    if (part.name != name) {
      continue;
    }

    Sam3ModelPartRuntimeStatus status;
    status.name = part.name;
    status.path = part.path;
    status.exists = std::filesystem::exists(part.path);

    if (!status.exists) {
      status.error = "missing";
      part_statuses_.push_back(std::move(status));
      if (require) {
        throw std::runtime_error("missing SAM3 model part: " + part.name + " -> " + part.path);
      }
      return;
    }

    try {
      BpuModel model(part.path);
      status.loaded = true;
      status.model_name = model.Name();
      status.compile_bpu_core_num = model.CompileBpuCoreNum();
      status.inputs = model.Inputs();
      status.outputs = model.Outputs();
      loaded_parts_.push_back(LoadedPart{part.name, std::move(model)});
      loaded_ = true;
    } catch (const std::exception& error) {
      status.error = error.what();
      if (require) {
        part_statuses_.push_back(std::move(status));
        throw;
      }
    }

    part_statuses_.push_back(std::move(status));
    return;
  }

  if (require) {
    throw std::runtime_error("unknown SAM3 model part: " + name);
  }
}

void Sam3Model::Reset() {
  loaded_parts_.clear();
  part_statuses_.clear();
  loaded_ = false;
}

const Sam3Config& Sam3Model::Config() const { return config_; }

bool Sam3Model::Loaded() const { return loaded_; }

const std::vector<Sam3ModelPartRuntimeStatus>& Sam3Model::PartStatuses() const { return part_statuses_; }

const BpuModel* Sam3Model::FindPart(const std::string& name) const {
  for (const auto& part : loaded_parts_) {
    if (part.name == name) {
      return &part.model;
    }
  }
  return nullptr;
}

}  // namespace sam_s600
