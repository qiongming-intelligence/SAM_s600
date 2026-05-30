#include "sam_s600/multiplex/multiplex_video_predictor.hpp"

#include <stdexcept>
#include <string>
#include <vector>

int main() {
  sam_s600::Sam3MultiplexVideoPredictor predictor{sam_s600::Sam3Model{}};

  sam_s600::VideoFrame frame;
  frame.pts_us = 123;
  frame.image.width = 16;
  frame.image.height = 16;
  frame.image.stride = 48;
  frame.image.format = sam_s600::PixelFormat::kRgb888;
  frame.image.data.resize(16 * 16 * 3);

  try {
    (void)predictor.Predict(std::vector<sam_s600::VideoFrame>{frame}, sam_s600::Sam3Prompt{});
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    if (message.find("missing required SAM3 model part: multiplex_detector") != std::string::npos) {
      return 0;
    }
    return 1;
  }

  return 1;
}
