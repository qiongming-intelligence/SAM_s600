#pragma once

#include <string>
#include <vector>

#include "sam_s600/sam3/sam3_config.hpp"

namespace sam_s600 {

struct Sam3Manifest {
  std::string name;
  std::string version;
  Sam3Config config;
};

struct Sam3ModelPart {
  std::string name;
  std::string path;
};

struct Sam3ModelPartStatus {
  Sam3ModelPart part;
  bool exists{false};
};

[[nodiscard]] std::vector<Sam3ModelPart> ListSam3ModelParts(const Sam3Config& config);
[[nodiscard]] std::vector<Sam3ModelPartStatus> CheckSam3ModelParts(const Sam3Config& config);

[[nodiscard]] Sam3Manifest LoadSam3Manifest(const std::string& path);

}  // namespace sam_s600
