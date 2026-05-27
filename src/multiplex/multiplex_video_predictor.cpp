#include "sam_s600/multiplex/multiplex_video_predictor.hpp"

#include <utility>

namespace sam_s600 {

Sam3MultiplexVideoPredictor::Sam3MultiplexVideoPredictor(Sam3Model model) : model_(std::move(model)) {}

Sam3VideoResult Sam3MultiplexVideoPredictor::Predict(const std::vector<VideoFrame>& frames, const Sam3Prompt& prompt) const {
  (void)frames;
  (void)prompt;
  return Sam3VideoResult{};
}

}  // namespace sam_s600
