#pragma once

#include "sam_s600/core/image.hpp"
#include "sam_s600/sam3/sam3_prompt.hpp"

namespace sam_s600 {

struct Sam3ProcessedInput {
  Image image;
  Sam3Prompt prompt;
};

class Sam3Processor {
 public:
  [[nodiscard]] Sam3ProcessedInput Process(const Image& image, const Sam3Prompt& prompt) const;
};

}  // namespace sam_s600
