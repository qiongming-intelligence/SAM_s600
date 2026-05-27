#include "sam_s600/core/profiler.hpp"

namespace sam_s600 {

void Profiler::Start(std::string name) {
  last_name_ = std::move(name);
  start_ = std::chrono::steady_clock::now();
}

void Profiler::Stop() {
  const auto end = std::chrono::steady_clock::now();
  last_duration_ms_ = std::chrono::duration<double, std::milli>(end - start_).count();
}

double Profiler::LastDurationMs() const { return last_duration_ms_; }

const std::string& Profiler::LastName() const { return last_name_; }

}  // namespace sam_s600
