#pragma once

#include "sam_s600/sam3/sam3_prompt.hpp"

namespace sam_s600 {

class Sam3GeometryEncoder {
 public:
  void Encode(const Sam3Prompt& prompt);
};

}  // namespace sam_s600
