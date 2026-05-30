#include "sam_s600/sam3/sam3_tokenizer.hpp"

#include <utility>

namespace sam_s600 {

Sam3Tokenizer::Sam3Tokenizer(std::string vocabulary_path) : vocabulary_path_(std::move(vocabulary_path)) {}

TokenizedPrompt Sam3Tokenizer::Tokenize(const std::string& text) const {
  TokenizedPrompt prompt;
  prompt.token_ids.reserve(text.size());
  prompt.attention_mask.reserve(text.size());
  for (unsigned char ch : text) {
    prompt.token_ids.push_back(static_cast<int>(ch));
    prompt.attention_mask.push_back(1);
  }
  return prompt;
}

const std::string& Sam3Tokenizer::VocabularyPath() const { return vocabulary_path_; }

}  // namespace sam_s600
