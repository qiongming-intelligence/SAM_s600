#include <exception>
#include <iostream>
#include <string>

#include "sam_s600/sam3/sam3_manifest.hpp"

namespace {

void PrintUsage(const char* app) {
  std::cout << "usage: " << app << " [--manifest models/manifests/sam3_image.yaml]\n";
}

std::string ManifestPath(int argc, char** argv, const std::string& fallback) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    if (arg == "--manifest") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--manifest requires a path");
      }
      return argv[++i];
    }
  }
  return fallback;
}

void PrintPart(const char* name, const std::string& path) {
  if (!path.empty()) {
    std::cout << "  " << name << ": " << path << '\n';
  }
}

int RunManifestCli(int argc, char** argv, const std::string& default_manifest, const std::string& mode) {
  const auto manifest = sam_s600::LoadSam3Manifest(ManifestPath(argc, argv, default_manifest));
  const auto& parts = manifest.config.model_parts;
  std::cout << mode << ": " << manifest.name << " (" << manifest.version << ")\n";
  std::cout << "SAM3 model parts:\n";
  PrintPart("image_encoder", parts.image_encoder);
  PrintPart("text_encoder", parts.text_encoder);
  PrintPart("geometry_encoder", parts.geometry_encoder);
  PrintPart("detector", parts.detector);
  PrintPart("mask_decoder", parts.mask_decoder);
  PrintPart("memory_encoder", parts.memory_encoder);
  PrintPart("tracker", parts.tracker);
  PrintPart("multiplex_detector", parts.multiplex_detector);
  PrintPart("multiplex_tracker", parts.multiplex_tracker);
  std::cout << "video: " << (manifest.config.enable_video ? "enabled" : "disabled") << '\n';
  std::cout << "multiplex: " << (manifest.config.enable_multiplex ? "enabled" : "disabled") << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return RunManifestCli(argc, argv, "models/manifests/sam3_full.yaml", "sam3_camera");
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
