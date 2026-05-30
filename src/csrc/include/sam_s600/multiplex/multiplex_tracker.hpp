#pragma once

#include "sam_s600/multiplex/multiplex_state.hpp"

namespace sam_s600 {

class Sam3MultiplexTracker {
 public:
  void Update(MultiplexState& state) const;
};

}  // namespace sam_s600
