#include "cli_common.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "sam_s600/sam3/sam3_manifest.hpp"

namespace {

void PrintUsage(const char* app) {
  std::cout << "usage: " << app << " [--manifest path] [--check-models]\n";
}

struct CliOptions {
  std::string manifest_path;
  bool check_models{false};
};

CliOptions ParseOptions(int argc, char** argv, const std::string& fallback) {
  CliOptions options{fallback, false};
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
      options.manifest_path = argv[++i];
      continue;
    }
    if (arg == "--check-models") {
      options.check_models = true;
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }
  return options;
}

void PrintPart(const char* name, const std::string& path) {
  if (!path.empty()) {
    std::cout << "  " << name << ": " << path << '\n';
  }
}

void PrintManifest(const sam_s600::Sam3Manifest& manifest, const std::string& mode) {
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
}

void PrintModelChecks(const sam_s600::Sam3Config& config) {
  const auto checks = sam_s600::CheckSam3ModelParts(config);
  for (const auto& item : checks) {
    std::cout << item.part.name << ": " << (item.exists ? "present" : "missing") << " -> " << item.part.path << '\n';
  }
}

}  // namespace

int RunManifestCli(int argc, char** argv, const std::string& default_manifest, const std::string& mode) {
  try {
    const auto options = ParseOptions(argc, argv, default_manifest);
    const auto manifest = sam_s600::LoadSam3Manifest(options.manifest_path);
    PrintManifest(manifest, mode);
    if (options.check_models) {
      PrintModelChecks(manifest.config);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
