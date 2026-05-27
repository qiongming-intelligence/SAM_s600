#pragma once

#include <vector>

#include "sam_s600/sam3/sam3_output.hpp"

namespace sam_s600 {

[[nodiscard]] inline std::vector<Sam3Object> Nms(const std::vector<Sam3Object>& objects, float threshold) {
  (void)threshold;
  return objects;
}

}  // namespace sam_s600
