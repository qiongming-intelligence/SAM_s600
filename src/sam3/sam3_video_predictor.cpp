#include "sam_s600/sam3/sam3_video_predictor.hpp"

#include <utility>

namespace sam_s600 {

Sam3VideoPredictor::Sam3VideoPredictor(Sam3Model model) : model_(std::move(model)) {}

Sam3VideoResult Sam3VideoPredictor::Predict(const std::vector<VideoFrame>& frames, const Sam3Prompt& prompt) const {
  (void)frames;
  (void)prompt;
  return Sam3VideoResult{};
}

}  // namespace sam_s600
