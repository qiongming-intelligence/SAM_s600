#pragma once

#include <vector>

#include "sam_s600/core/image.hpp"
#include "sam_s600/sam3/sam3_model.hpp"
#include "sam_s600/sam3/sam3_output.hpp"
#include "sam_s600/sam3/sam3_prompt.hpp"

namespace sam_s600 {

class Sam3VideoPredictor {
 public:
  explicit Sam3VideoPredictor(Sam3Model model);

  [[nodiscard]] Sam3VideoResult Predict(const std::vector<VideoFrame>& frames, const Sam3Prompt& prompt) const;

 private:
  Sam3Model model_;
};

}  // namespace sam_s600
