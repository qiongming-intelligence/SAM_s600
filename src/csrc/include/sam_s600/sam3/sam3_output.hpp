#pragma once

#include <string>
#include <vector>

#include "sam_s600/core/image.hpp"

namespace sam_s600 {

struct Sam3Mask {
  int width{0};
  int height{0};
  std::vector<float> logits;
};

struct Sam3Object {
  int id{0};
  float score{0.0F};
  float presence_score{0.0F};
  float x0{0.0F};
  float y0{0.0F};
  float x1{0.0F};
  float y1{0.0F};
  std::string label;
  Sam3Mask mask;
};

struct Sam3ImageResult {
  float presence_logit{0.0F};
  float presence_score{0.0F};
  std::vector<Sam3Object> objects;
  std::vector<Sam3Object> candidates;
};

struct Sam3VideoFrameResult {
  std::int64_t pts_us{0};
  std::vector<Sam3Object> objects;
};

struct Sam3VideoResult {
  std::vector<Sam3VideoFrameResult> frames;
};

}  // namespace sam_s600
