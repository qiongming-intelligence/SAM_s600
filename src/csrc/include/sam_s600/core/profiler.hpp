#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace sam_s600 {

class Profiler {
 public:
  void Start(std::string name);
  void Stop();
  [[nodiscard]] double LastDurationMs() const;
  [[nodiscard]] const std::string& LastName() const;

 private:
  std::string last_name_;
  std::chrono::steady_clock::time_point start_{};
  double last_duration_ms_{0.0};
};

struct BenchmarkResult {
  std::string name;
  double latency_ms{0.0};
  double fps{0.0};
  std::vector<double> bpu_loading;
  std::uint64_t hbmem_bytes{0};
};

}  // namespace sam_s600
