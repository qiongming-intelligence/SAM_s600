#pragma once

#include "sam_s600/sam3/sam3_output.hpp"

namespace sam_s600 {

[[nodiscard]] Sam3Mask ResizeMask(const Sam3Mask& mask, int width, int height);

}  // namespace sam_s600
