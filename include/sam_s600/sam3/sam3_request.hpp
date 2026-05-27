#pragma once

#include <string>

#include "sam_s600/sam3/sam3_prompt.hpp"

namespace sam_s600 {

enum class Sam3InputType {
  kUnspecified,
  kImage,
  kVideo,
  kRtsp,
  kCamera,
};

struct Sam3Request {
  Sam3InputType input_type{Sam3InputType::kUnspecified};
  std::string image_path;
  std::string video_path;
  std::string rtsp_url;
  std::string camera_device;
  Sam3Prompt prompt;
};

}  // namespace sam_s600
