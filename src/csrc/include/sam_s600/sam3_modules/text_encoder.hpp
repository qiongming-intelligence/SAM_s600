#pragma once

#include "sam_s600/sam3/sam3_tokenizer.hpp"

namespace sam_s600 {

class Sam3TextEncoder {
 public:
  void Encode(const TokenizedPrompt& prompt);
};

}  // namespace sam_s600
