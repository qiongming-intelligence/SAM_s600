#include "sam_s600/postprocess/mask_ops.hpp"

namespace sam_s600 {

Sam3Mask ResizeMask(const Sam3Mask& mask, int width, int height) {
  Sam3Mask resized = mask;
  resized.width = width;
  resized.height = height;
  return resized;
}

}  // namespace sam_s600
