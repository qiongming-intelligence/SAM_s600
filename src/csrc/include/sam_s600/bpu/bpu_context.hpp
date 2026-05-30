#pragma once

#include <string>

namespace sam_s600 {

struct BpuRuntimeOptions {
  int core_id{0};
  int thread_count{1};
  std::string profile_path;
};

class BpuContext {
 public:
  explicit BpuContext(BpuRuntimeOptions options = {});

  [[nodiscard]] const BpuRuntimeOptions& Options() const;
  [[nodiscard]] bool IsAvailable() const;
  [[nodiscard]] std::string RuntimeName() const;

 private:
  BpuRuntimeOptions options_;
};

}  // namespace sam_s600
