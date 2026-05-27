#pragma once

#include <vector>

#include "sam_s600/core/image.hpp"
#include "sam_s600/sam3/sam3_model.hpp"
#include "sam_s600/sam3/sam3_output.hpp"
#include "sam_s600/sam3/sam3_prompt.hpp"

namespace sam_s600 {

class Sam3ImagePredictor {
 public:
  explicit Sam3ImagePredictor(Sam3Model model);

  [[nodiscard]] Sam3ImageResult Predict(const Image& image, const Sam3Prompt& prompt) const;

 private:
  Sam3Model model_;
};

}  // namespace sam_s600
