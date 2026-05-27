#pragma once

#include <cstdint>
#include <vector>

namespace sam_s600 {

struct MultiplexObjectState {
  int object_id{0};
  std::int64_t last_pts_us{0};
};

struct MultiplexState {
  std::vector<MultiplexObjectState> objects;
};

}  // namespace sam_s600
