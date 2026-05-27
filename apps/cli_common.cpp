#include "cli_common.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sam_s600/sam3/sam3_manifest.hpp"
#include "sam_s600/sam3/sam3_request.hpp"

namespace {

void PrintUsage(const char* app) {
  std::cout << "usage: " << app
            << " [--manifest path] [--check-models]"
               " [--image path | --input path | --url rtsp-url | --camera device]"
               " [--text prompt] [--point x,y[,label]] [--box x0,y0,x1,y1]"
               " [--mask path] [--exemplar path]\n";
}

struct CliOptions {
  std::string manifest_path;
  bool check_models{false};
  sam_s600::Sam3Request request;
};

std::vector<std::string> Split(const std::string& value, char delimiter) {
  std::vector<std::string> parts;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    parts.push_back(item);
  }
  return parts;
}

float ParseFloat(const std::string& value, const std::string& field) {
  try {
    size_t pos = 0;
    const float parsed = std::stof(value, &pos);
    if (pos != value.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error("invalid " + field + ": " + value);
  }
}

int ParseInt(const std::string& value, const std::string& field) {
  try {
    size_t pos = 0;
    const int parsed = std::stoi(value, &pos);
    if (pos != value.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error("invalid " + field + ": " + value);
  }
}

sam_s600::Sam3PointPrompt ParsePoint(const std::string& value) {
  const auto parts = Split(value, ',');
  if (parts.size() != 2 && parts.size() != 3) {
    throw std::runtime_error("--point expects x,y or x,y,label");
  }
  sam_s600::Sam3PointPrompt point;
  point.x = ParseFloat(parts[0], "point x");
  point.y = ParseFloat(parts[1], "point y");
  if (parts.size() == 3) {
    point.label = ParseInt(parts[2], "point label");
  }
  return point;
}

sam_s600::Sam3BoxPrompt ParseBox(const std::string& value) {
  const auto parts = Split(value, ',');
  if (parts.size() != 4) {
    throw std::runtime_error("--box expects x0,y0,x1,y1");
  }
  sam_s600::Sam3BoxPrompt box;
  box.x0 = ParseFloat(parts[0], "box x0");
  box.y0 = ParseFloat(parts[1], "box y0");
  box.x1 = ParseFloat(parts[2], "box x1");
  box.y1 = ParseFloat(parts[3], "box y1");
  return box;
}

std::string RequiredValue(int& index, int argc, char** argv, const std::string& flag) {
  if (index + 1 >= argc) {
    throw std::runtime_error(flag + " requires a value");
  }
  return argv[++index];
}

void AddType(sam_s600::Sam3Prompt& prompt, sam_s600::Sam3PromptType type) {
  prompt.types.push_back(type);
}

CliOptions ParseOptions(int argc, char** argv, const std::string& fallback) {
  CliOptions options;
  options.manifest_path = fallback;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    if (arg == "--manifest") {
      options.manifest_path = RequiredValue(i, argc, argv, arg);
      continue;
    }
    if (arg == "--check-models") {
      options.check_models = true;
      continue;
    }
    if (arg == "--image") {
      options.request.input_type = sam_s600::Sam3InputType::kImage;
      options.request.image_path = RequiredValue(i, argc, argv, arg);
      continue;
    }
    if (arg == "--input") {
      options.request.input_type = sam_s600::Sam3InputType::kVideo;
      options.request.video_path = RequiredValue(i, argc, argv, arg);
      continue;
    }
    if (arg == "--url") {
      options.request.input_type = sam_s600::Sam3InputType::kRtsp;
      options.request.rtsp_url = RequiredValue(i, argc, argv, arg);
      continue;
    }
    if (arg == "--camera") {
      options.request.input_type = sam_s600::Sam3InputType::kCamera;
      options.request.camera_device = RequiredValue(i, argc, argv, arg);
      continue;
    }
    if (arg == "--text") {
      options.request.prompt.text = RequiredValue(i, argc, argv, arg);
      AddType(options.request.prompt, sam_s600::Sam3PromptType::kText);
      continue;
    }
    if (arg == "--point") {
      options.request.prompt.points.push_back(ParsePoint(RequiredValue(i, argc, argv, arg)));
      AddType(options.request.prompt, sam_s600::Sam3PromptType::kPoint);
      continue;
    }
    if (arg == "--box") {
      options.request.prompt.boxes.push_back(ParseBox(RequiredValue(i, argc, argv, arg)));
      AddType(options.request.prompt, sam_s600::Sam3PromptType::kBox);
      continue;
    }
    if (arg == "--mask") {
      options.request.prompt.mask_path = RequiredValue(i, argc, argv, arg);
      AddType(options.request.prompt, sam_s600::Sam3PromptType::kMask);
      continue;
    }
    if (arg == "--exemplar") {
      options.request.prompt.exemplar_path = RequiredValue(i, argc, argv, arg);
      AddType(options.request.prompt, sam_s600::Sam3PromptType::kExemplar);
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

const char* InputTypeName(sam_s600::Sam3InputType type) {
  switch (type) {
    case sam_s600::Sam3InputType::kImage:
      return "image";
    case sam_s600::Sam3InputType::kVideo:
      return "video";
    case sam_s600::Sam3InputType::kRtsp:
      return "rtsp";
    case sam_s600::Sam3InputType::kCamera:
      return "camera";
    case sam_s600::Sam3InputType::kUnspecified:
      return "unspecified";
  }
  return "unspecified";
}

void PrintRequest(const sam_s600::Sam3Request& request) {
  std::cout << "input: " << InputTypeName(request.input_type) << '\n';
  PrintPart("image", request.image_path);
  PrintPart("video", request.video_path);
  PrintPart("rtsp_url", request.rtsp_url);
  PrintPart("camera", request.camera_device);

  const auto& prompt = request.prompt;
  if (!prompt.text.empty()) {
    std::cout << "text: " << prompt.text << '\n';
  }
  for (const auto& point : prompt.points) {
    std::cout << "point: " << point.x << ',' << point.y << ',' << point.label << '\n';
  }
  for (const auto& box : prompt.boxes) {
    std::cout << "box: " << box.x0 << ',' << box.y0 << ',' << box.x1 << ',' << box.y1 << '\n';
  }
  PrintPart("mask", prompt.mask_path);
  PrintPart("exemplar", prompt.exemplar_path);
}

}  // namespace

int RunManifestCli(int argc, char** argv, const std::string& default_manifest, const std::string& mode) {
  try {
    const auto options = ParseOptions(argc, argv, default_manifest);
    const auto manifest = sam_s600::LoadSam3Manifest(options.manifest_path);
    PrintManifest(manifest, mode);
    PrintRequest(options.request);
    if (options.check_models) {
      PrintModelChecks(manifest.config);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
