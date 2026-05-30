#include "sam_s600/cli/manifest_cli.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef SAM_S600_SOURCE_DIR
#error "SAM_S600_SOURCE_DIR is required"
#endif

int main() {
  const auto frame_path = std::filesystem::temp_directory_path() / "sam3_video_cli_smoke_frame.bin";
  {
    std::ofstream frame(frame_path, std::ios::binary);
    const std::vector<char> bytes(16 * 16 * 3, '\0');
    frame.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  const std::string manifest = std::string(SAM_S600_SOURCE_DIR) + "/configs/manifests/sam3_video.yaml";
  const std::string input = frame_path.string();
  const char* argv[] = {"sam3_video", "--manifest", manifest.c_str(), "--input", input.c_str(), "--run"};
  const int result = RunManifestCli(6, const_cast<char**>(argv), manifest, "sam3_video");

  std::filesystem::remove(frame_path);
  return result == 1 ? 0 : 1;
}
