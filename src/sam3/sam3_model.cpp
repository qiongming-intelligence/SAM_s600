#include "sam_s600/sam3/sam3_model.hpp"

#include <utility>

namespace sam_s600 {

Sam3Model::Sam3Model(Sam3Config config, BpuRuntimeOptions runtime)
    : config_(std::move(config)), context_(std::move(runtime)) {}

void Sam3Model::Load() { loaded_ = true; }

const Sam3Config& Sam3Model::Config() const { return config_; }

bool Sam3Model::Loaded() const { return loaded_; }

}  // namespace sam_s600
