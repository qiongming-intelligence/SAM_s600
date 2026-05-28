#include "sam_s600/sam3/sam3_manifest.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace sam_s600 {
namespace {

std::string Trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::pair<std::string, std::string> ParseKeyValue(const std::string& line) {
  const auto colon = line.find(':');
  if (colon == std::string::npos) {
    return {};
  }
  return {Trim(line.substr(0, colon)), Trim(line.substr(colon + 1))};
}

void ApplyPart(Sam3ModelPartPaths& parts, const std::string& key, const std::string& value) {
  if (key == "image_encoder") {
    parts.image_encoder = value;
  } else if (key == "text_encoder") {
    parts.text_encoder = value;
  } else if (key == "geometry_encoder") {
    parts.geometry_encoder = value;
  } else if (key == "detector") {
    parts.detector = value;
  } else if (key == "detector_taps") {
    parts.detector_taps = value;
  } else if (key == "mask_decoder") {
    parts.mask_decoder = value;
  } else if (key == "mask_decoder_pre_norm") {
    parts.mask_decoder_pre_norm = value;
  } else if (key == "mask_decoder_post_norm") {
    parts.mask_decoder_post_norm = value;
  } else if (key == "mask_decoder_pixel_mid") {
    parts.mask_decoder_pixel_mid = value;
  } else if (key == "mask_decoder_pixel2_post_norm") {
    parts.mask_decoder_pixel2_post_norm = value;
  } else if (key == "memory_encoder") {
    parts.memory_encoder = value;
  } else if (key == "tracker") {
    parts.tracker = value;
  } else if (key == "multiplex_detector") {
    parts.multiplex_detector = value;
  } else if (key == "multiplex_tracker") {
    parts.multiplex_tracker = value;
  }
}

void MergeConfig(Sam3Config& dst, const Sam3Config& src) {
  auto& d = dst.model_parts;
  const auto& s = src.model_parts;
  if (!s.image_encoder.empty()) d.image_encoder = s.image_encoder;
  if (!s.text_encoder.empty()) d.text_encoder = s.text_encoder;
  if (!s.geometry_encoder.empty()) d.geometry_encoder = s.geometry_encoder;
  if (!s.detector.empty()) d.detector = s.detector;
  if (!s.detector_taps.empty()) d.detector_taps = s.detector_taps;
  if (!s.mask_decoder.empty()) d.mask_decoder = s.mask_decoder;
  if (!s.mask_decoder_pre_norm.empty()) d.mask_decoder_pre_norm = s.mask_decoder_pre_norm;
  if (!s.mask_decoder_post_norm.empty()) d.mask_decoder_post_norm = s.mask_decoder_post_norm;
  if (!s.mask_decoder_pixel_mid.empty()) d.mask_decoder_pixel_mid = s.mask_decoder_pixel_mid;
  if (!s.mask_decoder_pixel2_post_norm.empty()) d.mask_decoder_pixel2_post_norm = s.mask_decoder_pixel2_post_norm;
  if (!s.memory_encoder.empty()) d.memory_encoder = s.memory_encoder;
  if (!s.tracker.empty()) d.tracker = s.tracker;
  if (!s.multiplex_detector.empty()) d.multiplex_detector = s.multiplex_detector;
  if (!s.multiplex_tracker.empty()) d.multiplex_tracker = s.multiplex_tracker;
  dst.enable_video = dst.enable_video || src.enable_video;
  dst.enable_multiplex = dst.enable_multiplex || src.enable_multiplex;
}

Sam3Manifest LoadSam3ManifestImpl(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open SAM3 manifest: " + path.string());
  }

  Sam3Manifest manifest;
  bool in_parts = false;
  bool in_include = false;
  std::string line;

  while (std::getline(input, line)) {
    const auto trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    if (trimmed == "parts:") {
      in_parts = true;
      in_include = false;
      continue;
    }
    if (trimmed == "include:") {
      in_parts = false;
      in_include = true;
      continue;
    }

    if (in_include && trimmed.rfind("- ", 0) == 0) {
      const auto include_path = path.parent_path() / Trim(trimmed.substr(2));
      auto included = LoadSam3ManifestImpl(include_path);
      MergeConfig(manifest.config, included.config);
      if (manifest.name.empty()) {
        manifest.name = path.stem().string();
      }
      continue;
    }

    auto [key, value] = ParseKeyValue(trimmed);
    if (key.empty()) {
      continue;
    }

    if (in_parts) {
      ApplyPart(manifest.config.model_parts, key, value);
      if (key == "memory_encoder" || key == "tracker") {
        manifest.config.enable_video = true;
      }
      if (key == "multiplex_detector" || key == "multiplex_tracker") {
        manifest.config.enable_multiplex = true;
      }
    } else if (key == "name") {
      manifest.name = value;
    } else if (key == "version") {
      manifest.version = value;
    }
  }

  if (manifest.name.empty()) {
    manifest.name = path.stem().string();
  }
  return manifest;
}

}  // namespace

Sam3Manifest LoadSam3Manifest(const std::string& path) {
  return LoadSam3ManifestImpl(std::filesystem::path(path));
}

}  // namespace sam_s600


namespace sam_s600 {

std::vector<Sam3ModelPart> ListSam3ModelParts(const Sam3Config& config) {
  const auto& parts = config.model_parts;
  std::vector<Sam3ModelPart> result;
  const auto append = [&result](const std::string& name, const std::string& path) {
    if (!path.empty()) {
      result.push_back(Sam3ModelPart{name, path});
    }
  };

  append("image_encoder", parts.image_encoder);
  append("text_encoder", parts.text_encoder);
  append("geometry_encoder", parts.geometry_encoder);
  append("detector", parts.detector);
  append("detector_taps", parts.detector_taps);
  append("mask_decoder", parts.mask_decoder);
  append("mask_decoder_pre_norm", parts.mask_decoder_pre_norm);
  append("mask_decoder_post_norm", parts.mask_decoder_post_norm);
  append("mask_decoder_pixel_mid", parts.mask_decoder_pixel_mid);
  append("mask_decoder_pixel2_post_norm", parts.mask_decoder_pixel2_post_norm);
  append("memory_encoder", parts.memory_encoder);
  append("tracker", parts.tracker);
  append("multiplex_detector", parts.multiplex_detector);
  append("multiplex_tracker", parts.multiplex_tracker);
  return result;
}

std::vector<Sam3ModelPartStatus> CheckSam3ModelParts(const Sam3Config& config) {
  std::vector<Sam3ModelPartStatus> result;
  for (const auto& part : ListSam3ModelParts(config)) {
    result.push_back(Sam3ModelPartStatus{part, std::filesystem::exists(part.path)});
  }
  return result;
}

}  // namespace sam_s600
