#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam_s600 {

enum class PixelFormat {
  kUnknown,
  kRgb888,
  kBgr888,
  kNv12,
};

struct Image {
  int width{0};
  int height{0};
  int stride{0};
  PixelFormat format{PixelFormat::kUnknown};
  std::vector<std::uint8_t> data;
};

struct VideoFrame {
  Image image;
  std::int64_t pts_us{0};
};

}  // namespace sam_s600
