#pragma once

#include <string>
#include <vector>

namespace sam_s600 {

struct TokenizedPrompt {
  std::vector<int> token_ids;
  std::vector<int> attention_mask;
};

class Sam3Tokenizer {
 public:
  explicit Sam3Tokenizer(std::string vocabulary_path = {});

  [[nodiscard]] TokenizedPrompt Tokenize(const std::string& text) const;
  [[nodiscard]] const std::string& VocabularyPath() const;

 private:
  std::string vocabulary_path_;
};

}  // namespace sam_s600
