#include "sam_s600/sam3/sam3_image_predictor.hpp"

#include <stdexcept>
#include <string>

int main() {
  sam_s600::Sam3ImagePredictor predictor{sam_s600::Sam3Model{}};

  sam_s600::Image image;
  image.width = 16;
  image.height = 16;
  image.stride = 48;
  image.format = sam_s600::PixelFormat::kRgb888;
  image.data.resize(16 * 16 * 3);

  try {
    (void)predictor.Predict(image, sam_s600::Sam3Prompt{});
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    if (message.find("missing required SAM3 model part: image_encoder") != std::string::npos ||
        message.find("missing required SAM3 model part: detector") != std::string::npos) {
      return 0;
    }
    return 1;
  }

  return 1;
}
