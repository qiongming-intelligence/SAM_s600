#include "sam_s600/sam3/sam3_image_predictor.hpp"

#include <utility>

namespace sam_s600 {

Sam3ImagePredictor::Sam3ImagePredictor(Sam3Model model) : model_(std::move(model)) {}

Sam3ImageResult Sam3ImagePredictor::Predict(const Image& image, const Sam3Prompt& prompt) const {
  (void)image;
  (void)prompt;
  return Sam3ImageResult{};
}

}  // namespace sam_s600
