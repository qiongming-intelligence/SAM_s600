#pragma once

#include <string>

namespace sam_s600 {

struct Sam3ModelPartPaths {
  std::string image_encoder;
  std::string text_encoder;
  std::string geometry_encoder;
  std::string detector;
  std::string detector_taps;
  std::string detector_bridge_taps;
  std::string detector_image_bridge_taps;
  std::string detector_text_bridge_tap;
  std::string detector_geometry_bridge_taps;
  std::string detector_encoder_hidden_tap;
  std::string mask_decoder;
  std::string mask_decoder_pre_norm;
  std::string mask_decoder_post_norm;
  std::string mask_decoder_pixel_mid;
  std::string mask_decoder_pixel2_post_norm;
  std::string memory_encoder;
  std::string tracker;
  std::string multiplex_detector;
  std::string multiplex_tracker;
};

struct Sam3Config {
  Sam3ModelPartPaths model_parts;
  int image_width{1008};
  int image_height{1008};
  int max_text_tokens{256};
  int max_objects{256};
  bool enable_video{false};
  bool enable_multiplex{false};
};

}  // namespace sam_s600
