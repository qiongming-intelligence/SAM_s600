#pragma once

#include <string>

#include "sam_s600/sam3/sam3_config.hpp"

namespace sam_s600 {

struct Sam3Manifest {
  std::string name;
  std::string version;
  Sam3Config config;
};

[[nodiscard]] Sam3Manifest LoadSam3Manifest(const std::string& path);

}  // namespace sam_s600
