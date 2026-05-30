#pragma once

#include "sam_s600/core/image.hpp"
#include "sam_s600/sam3/sam3_output.hpp"

namespace sam_s600 {

[[nodiscard]] inline Image DrawMasks(const Image& image, const Sam3ImageResult& result) {
  (void)result;
  return image;
}

}  // namespace sam_s600
