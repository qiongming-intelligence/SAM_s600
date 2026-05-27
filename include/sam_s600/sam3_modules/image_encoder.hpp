#pragma once

#include "sam_s600/core/image.hpp"

namespace sam_s600 {

class Sam3ImageEncoder {
 public:
  void Encode(const Image& image);
};

}  // namespace sam_s600
