#pragma once

#include <string>
#include <vector>

namespace sam_s600 {

enum class Sam3PromptType {
  kText,
  kPoint,
  kBox,
  kMask,
  kExemplar,
};

struct Sam3PointPrompt {
  float x{0.0F};
  float y{0.0F};
  int label{1};
};

struct Sam3BoxPrompt {
  float x0{0.0F};
  float y0{0.0F};
  float x1{0.0F};
  float y1{0.0F};
};

struct Sam3Prompt {
  Sam3PromptType type{Sam3PromptType::kText};
  std::string text;
  std::vector<Sam3PointPrompt> points;
  std::vector<Sam3BoxPrompt> boxes;
  std::string mask_path;
  std::string exemplar_path;
};

}  // namespace sam_s600
