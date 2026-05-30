#include "sam_s600/sam3/sam3_processor.hpp"

namespace sam_s600 {

Sam3ProcessedInput Sam3Processor::Process(const Image& image, const Sam3Prompt& prompt) const {
  return Sam3ProcessedInput{image, prompt};
}

}  // namespace sam_s600
