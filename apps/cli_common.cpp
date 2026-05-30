#include "cli_common.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "sam_s600/bpu/bpu_model.hpp"
#include "sam_s600/core/tensor.hpp"
#include "sam_s600/multiplex/multiplex_video_predictor.hpp"
#include "sam_s600/sam3/sam3_image_predictor.hpp"
#include "sam_s600/sam3/sam3_manifest.hpp"
#include "sam_s600/sam3/sam3_model.hpp"
#include "sam_s600/sam3/sam3_request.hpp"
#include "sam_s600/sam3/sam3_video_predictor.hpp"

namespace {

void PrintUsage(const char* app) {
  std::cout << "usage: " << app
            << " [--manifest path] [--check-models] [--load-parts] [--require-all] [--run] [--serve]"
               " [--image path | --input raw-frame.bin|frames/ | --url rtsp-url | --camera device]"
               " [--text prompt] [--point x,y[,label]] [--box x0,y0,x1,y1]"
               " [--mask path] [--exemplar path] [--inspect-part hbm-path]\n";
}

struct CliOptions {
  std::string manifest_path;
  bool check_models{false};
  bool load_parts{false};
  bool require_all{false};
  bool run{false};
  bool serve{false};
  std::string inspect_part_path;
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
    if (arg == "--load-parts") {
      options.load_parts = true;
      continue;
    }
    if (arg == "--require-all") {
      options.require_all = true;
      continue;
    }
    if (arg == "--run") {
      options.run = true;
      continue;
    }
    if (arg == "--serve") {
      options.serve = true;
      continue;
    }
    if (arg == "--inspect-part") {
      options.inspect_part_path = RequiredValue(i, argc, argv, arg);
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
  PrintPart("detector_taps", parts.detector_taps);
  PrintPart("detector_bridge_taps", parts.detector_bridge_taps);
  PrintPart("detector_image_bridge_taps", parts.detector_image_bridge_taps);
  PrintPart("detector_text_bridge_tap", parts.detector_text_bridge_tap);
  PrintPart("detector_geometry_bridge_taps", parts.detector_geometry_bridge_taps);
  PrintPart("detector_encoder_hidden_tap", parts.detector_encoder_hidden_tap);
  PrintPart("mask_decoder", parts.mask_decoder);
  PrintPart("mask_decoder_pre_norm", parts.mask_decoder_pre_norm);
  PrintPart("mask_decoder_post_norm", parts.mask_decoder_post_norm);
  PrintPart("mask_decoder_pixel_mid", parts.mask_decoder_pixel_mid);
  PrintPart("mask_decoder_pixel2_post_norm", parts.mask_decoder_pixel2_post_norm);
  PrintPart("memory_encoder", parts.memory_encoder);
  PrintPart("tracker", parts.tracker);
  PrintPart("multiplex_detector", parts.multiplex_detector);
  PrintPart("multiplex_tracker", parts.multiplex_tracker);
  std::cout << "video: " << (manifest.config.enable_video ? "enabled" : "disabled") << '\n';
  std::cout << "multiplex: " << (manifest.config.enable_multiplex ? "enabled" : "disabled") << '\n';
}


const char* TensorTypeName(sam_s600::TensorDataType type) {
  switch (type) {
    case sam_s600::TensorDataType::kInt4:
      return "s4";
    case sam_s600::TensorDataType::kUint4:
      return "u4";
    case sam_s600::TensorDataType::kInt8:
      return "s8";
    case sam_s600::TensorDataType::kUint8:
      return "u8";
    case sam_s600::TensorDataType::kFloat16:
      return "f16";
    case sam_s600::TensorDataType::kInt16:
      return "s16";
    case sam_s600::TensorDataType::kUint16:
      return "u16";
    case sam_s600::TensorDataType::kFloat32:
      return "f32";
    case sam_s600::TensorDataType::kInt32:
      return "s32";
    case sam_s600::TensorDataType::kUint32:
      return "u32";
    case sam_s600::TensorDataType::kFloat64:
      return "f64";
    case sam_s600::TensorDataType::kInt64:
      return "s64";
    case sam_s600::TensorDataType::kUint64:
      return "u64";
    case sam_s600::TensorDataType::kBool8:
      return "bool8";
    case sam_s600::TensorDataType::kUnknown:
      return "unknown";
  }
  return "unknown";
}

void PrintShape(const sam_s600::TensorShape& shape) {
  std::cout << '(';
  for (std::size_t i = 0; i < shape.dims.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << shape.dims[i];
  }
  std::cout << ')';
}

void PrintTensorList(const char* title, const std::vector<sam_s600::TensorInfo>& tensors) {
  std::cout << title << ":\n";
  for (const auto& tensor : tensors) {
    std::cout << "  " << tensor.name << " shape=";
    PrintShape(tensor.shape);
    std::cout << " dtype=" << TensorTypeName(tensor.dtype) << " bytes=" << tensor.byte_size << '\n';
  }
}

void InspectModelPart(const std::string& path) {
  sam_s600::BpuModel model(path);
  std::cout << "inspected_part: " << model.Path() << '\n';
  std::cout << "model_name: " << model.Name() << '\n';
  std::cout << "compile_bpu_core_num: " << model.CompileBpuCoreNum() << '\n';
  PrintTensorList("inputs", model.Inputs());
  PrintTensorList("outputs", model.Outputs());
}


void PrintPartStatuses(const std::vector<sam_s600::Sam3ModelPartRuntimeStatus>& statuses) {
  std::cout << "SAM3 partition load status:\n";
  for (const auto& status : statuses) {
    std::cout << "  " << status.name << ": ";
    if (status.loaded) {
      std::cout << "loaded";
      if (!status.model_name.empty()) {
        std::cout << " model=" << status.model_name;
      }
      std::cout << " core_num=" << status.compile_bpu_core_num;
    } else if (!status.exists) {
      std::cout << "missing";
    } else {
      std::cout << "failed";
      if (!status.error.empty()) {
        std::cout << " error=" << status.error;
      }
    }
    std::cout << " -> " << status.path << '\n';
  }
}

void LoadModelParts(const sam_s600::Sam3Manifest& manifest, bool require_all) {
  sam_s600::Sam3Model model(manifest.config);
  model.Load(require_all);
  PrintPartStatuses(model.PartStatuses());
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

sam_s600::Image ReadRawImageBytes(const std::string& path, int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("SAM3 input dimensions are invalid");
  }

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open image input: " + path);
  }

  sam_s600::Image image;
  image.format = sam_s600::PixelFormat::kUnknown;
  image.width = width;
  image.height = height;
  image.data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  if (image.data.empty()) {
    throw std::runtime_error("image input is empty: " + path);
  }
  image.stride = static_cast<int>(image.data.size());
  return image;
}

sam_s600::VideoFrame ReadRawVideoFrame(const std::string& path, int width, int height, std::int64_t pts_us) {
  sam_s600::VideoFrame frame;
  frame.pts_us = pts_us;
  frame.image = ReadRawImageBytes(path, width, height);
  return frame;
}

std::vector<sam_s600::VideoFrame> ReadRawVideoFrames(const std::string& path, int width, int height) {
  std::vector<std::string> frame_paths;
  const std::filesystem::path input_path(path);
  if (std::filesystem::is_directory(input_path)) {
    for (const auto& entry : std::filesystem::directory_iterator(input_path)) {
      if (entry.is_regular_file()) {
        frame_paths.push_back(entry.path().string());
      }
    }
    std::sort(frame_paths.begin(), frame_paths.end());
    if (frame_paths.empty()) {
      throw std::runtime_error("video input directory is empty: " + path);
    }
  } else {
    frame_paths.push_back(path);
  }

  std::vector<sam_s600::VideoFrame> frames;
  frames.reserve(frame_paths.size());
  for (std::size_t i = 0; i < frame_paths.size(); ++i) {
    frames.push_back(ReadRawVideoFrame(frame_paths[i], width, height, static_cast<std::int64_t>(i)));
  }
  return frames;
}

void PrintImageResult(const sam_s600::Sam3ImageResult& result) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "presence_logit: " << result.presence_logit << '\n';
  std::cout << "presence_score: " << result.presence_score << '\n';
  std::cout << "result_objects: " << result.objects.size() << '\n';
  for (std::size_t i = 0; i < result.objects.size(); ++i) {
    const auto& object = result.objects[i];
    std::cout << "object[" << i << "]: id=" << object.id << " score=" << object.score
              << " box=" << object.x0 << ',' << object.y0 << ',' << object.x1 << ',' << object.y1 << '\n';
  }
  std::cout << "candidate_objects: " << result.candidates.size() << '\n';
  for (std::size_t i = 0; i < result.candidates.size(); ++i) {
    const auto& object = result.candidates[i];
    std::cout << "candidate[" << i << "]: id=" << object.id << " score=" << object.score
              << " box=" << object.x0 << ',' << object.y0 << ',' << object.x1 << ',' << object.y1 << '\n';
  }
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch)
              << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

void SkipJsonWhitespace(const std::string& text, std::size_t& pos) {
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
    ++pos;
  }
}

int JsonHexDigit(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return ch - 'a' + 10;
  }
  if (ch >= 'A' && ch <= 'F') {
    return ch - 'A' + 10;
  }
  return -1;
}

std::uint32_t ParseJsonUnicodeEscape(const std::string& text, std::size_t& pos) {
  if (pos + 4 > text.size()) {
    throw std::runtime_error("invalid JSON unicode escape");
  }
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    const int digit = JsonHexDigit(text[pos++]);
    if (digit < 0) {
      throw std::runtime_error("invalid JSON unicode escape");
    }
    value = (value << 4U) | static_cast<std::uint32_t>(digit);
  }
  return value;
}

void AppendUtf8(std::string& out, std::uint32_t codepoint) {
  if (codepoint <= 0x7FU) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FFU) {
    out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else if (codepoint <= 0xFFFFU) {
    out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else if (codepoint <= 0x10FFFFU) {
    out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else {
    throw std::runtime_error("invalid JSON unicode escape");
  }
}

std::string ParseJsonStringValue(const std::string& text, std::size_t& pos) {
  SkipJsonWhitespace(text, pos);
  if (pos >= text.size() || text[pos] != '"') {
    throw std::runtime_error("expected JSON string");
  }
  ++pos;
  std::string out;
  while (pos < text.size()) {
    const char ch = text[pos++];
    if (ch == '"') {
      return out;
    }
    if (ch != '\\') {
      out.push_back(ch);
      continue;
    }
    if (pos >= text.size()) {
      throw std::runtime_error("invalid JSON escape");
    }
    const char esc = text[pos++];
    switch (esc) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case 'u': {
        std::uint32_t codepoint = ParseJsonUnicodeEscape(text, pos);
        if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
          if (pos + 2 > text.size() || text[pos] != '\\' || text[pos + 1] != 'u') {
            throw std::runtime_error("invalid JSON unicode surrogate pair");
          }
          pos += 2;
          const std::uint32_t low = ParseJsonUnicodeEscape(text, pos);
          if (low < 0xDC00U || low > 0xDFFFU) {
            throw std::runtime_error("invalid JSON unicode surrogate pair");
          }
          codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
        } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
          throw std::runtime_error("invalid JSON unicode surrogate pair");
        }
        AppendUtf8(out, codepoint);
        break;
      }
      default: throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

bool FindJsonKeyValue(const std::string& text, const std::string& key, std::size_t& value_pos) {
  std::size_t pos = 0;
  while (pos < text.size()) {
    SkipJsonWhitespace(text, pos);
    if (pos >= text.size()) {
      return false;
    }
    if (text[pos] != '"') {
      ++pos;
      continue;
    }
    auto parsed_key = ParseJsonStringValue(text, pos);
    SkipJsonWhitespace(text, pos);
    if (pos >= text.size() || text[pos] != ':') {
      continue;
    }
    ++pos;
    if (parsed_key == key) {
      value_pos = pos;
      return true;
    }
  }
  return false;
}

bool ParseOptionalJsonString(const std::string& text, const std::string& key, std::string& value) {
  std::size_t pos = 0;
  if (!FindJsonKeyValue(text, key, pos)) {
    return false;
  }
  value = ParseJsonStringValue(text, pos);
  return true;
}

std::vector<float> ParseJsonNumberArray(const std::string& text, const std::string& key, std::size_t min_count, std::size_t max_count) {
  std::size_t pos = 0;
  if (!FindJsonKeyValue(text, key, pos)) {
    return {};
  }
  SkipJsonWhitespace(text, pos);
  if (pos >= text.size() || text[pos] != '[') {
    throw std::runtime_error("expected JSON array for key: " + key);
  }
  ++pos;
  std::vector<float> values;
  while (pos < text.size()) {
    SkipJsonWhitespace(text, pos);
    if (pos < text.size() && text[pos] == ']') {
      ++pos;
      break;
    }
    size_t parsed = 0;
    const float value = std::stof(text.substr(pos), &parsed);
    values.push_back(value);
    pos += parsed;
    SkipJsonWhitespace(text, pos);
    if (pos < text.size() && text[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < text.size() && text[pos] == ']') {
      ++pos;
      break;
    }
    throw std::runtime_error("expected comma or ] in array for key: " + key);
  }
  if (values.size() < min_count || values.size() > max_count) {
    throw std::runtime_error("invalid array length for key: " + key);
  }
  return values;
}

sam_s600::Sam3Request ParseJsonLineRequest(const std::string& line) {
  sam_s600::Sam3Request request;
  request.input_type = sam_s600::Sam3InputType::kImage;
  if (!ParseOptionalJsonString(line, "image", request.image_path) || request.image_path.empty()) {
    throw std::runtime_error("JSON request requires non-empty image");
  }
  if (ParseOptionalJsonString(line, "text", request.prompt.text) && !request.prompt.text.empty()) {
    AddType(request.prompt, sam_s600::Sam3PromptType::kText);
  }
  const auto point_values = ParseJsonNumberArray(line, "point", 2, 3);
  if (!point_values.empty()) {
    sam_s600::Sam3PointPrompt point;
    point.x = point_values[0];
    point.y = point_values[1];
    if (point_values.size() == 3) {
      point.label = static_cast<int>(point_values[2]);
    }
    request.prompt.points.push_back(point);
    AddType(request.prompt, sam_s600::Sam3PromptType::kPoint);
  }
  const auto box_values = ParseJsonNumberArray(line, "box", 4, 4);
  if (!box_values.empty()) {
    sam_s600::Sam3BoxPrompt box;
    box.x0 = box_values[0];
    box.y0 = box_values[1];
    box.x1 = box_values[2];
    box.y1 = box_values[3];
    request.prompt.boxes.push_back(box);
    AddType(request.prompt, sam_s600::Sam3PromptType::kBox);
  }
  return request;
}

void WriteJsonObject(std::ostream& out, const sam_s600::Sam3Object& object) {
  out << "{\"id\":" << object.id
      << ",\"score\":" << object.score
      << ",\"presence_score\":" << object.presence_score
      << ",\"box\":[" << object.x0 << ',' << object.y0 << ',' << object.x1 << ',' << object.y1 << ']';
  if (!object.label.empty()) {
    out << ",\"label\":\"" << JsonEscape(object.label) << '"';
  }
  out << '}';
}

void WriteJsonObjectArray(std::ostream& out, const std::vector<sam_s600::Sam3Object>& objects) {
  out << '[';
  for (std::size_t i = 0; i < objects.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    WriteJsonObject(out, objects[i]);
  }
  out << ']';
}

void PrintJsonImageResult(const sam_s600::Sam3ImageResult& result, double elapsed_ms) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\"ok\":true"
            << ",\"elapsed_ms\":" << elapsed_ms
            << ",\"presence_logit\":" << result.presence_logit
            << ",\"presence_score\":" << result.presence_score
            << ",\"objects\":";
  WriteJsonObjectArray(std::cout, result.objects);
  std::cout << ",\"candidates\":";
  WriteJsonObjectArray(std::cout, result.candidates);
  std::cout << "}\n";
  std::cout.flush();
}

void PrintJsonError(const std::string& message) {
  std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(message) << "\"}\n";
  std::cout.flush();
}

class StdoutToStderrGuard {
 public:
  StdoutToStderrGuard() {
    std::cout.flush();
    std::fflush(stdout);
    saved_stdout_ = dup(STDOUT_FILENO);
    if (saved_stdout_ < 0) {
      throw std::runtime_error("failed to duplicate stdout");
    }
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
      close(saved_stdout_);
      saved_stdout_ = -1;
      throw std::runtime_error("failed to redirect stdout to stderr");
    }
  }

  StdoutToStderrGuard(const StdoutToStderrGuard&) = delete;
  StdoutToStderrGuard& operator=(const StdoutToStderrGuard&) = delete;

  ~StdoutToStderrGuard() {
    if (saved_stdout_ >= 0) {
      std::fflush(stdout);
      (void)dup2(saved_stdout_, STDOUT_FILENO);
      close(saved_stdout_);
    }
  }

 private:
  int saved_stdout_{-1};
};

void CheckRequiredModelParts(const sam_s600::Sam3Manifest& manifest, bool require_all) {
  if (!require_all) {
    return;
  }
  for (const auto& item : sam_s600::CheckSam3ModelParts(manifest.config)) {
    if (!item.exists) {
      throw std::runtime_error("missing SAM3 model part: " + item.part.name + " -> " + item.part.path);
    }
  }
}

std::vector<std::string> ImageServeResidentParts(const sam_s600::Sam3Manifest& manifest) {
  const auto& parts = manifest.config.model_parts;
  std::vector<std::string> names;
  const auto append = [&](const char* name, const std::string& path) {
    if (!path.empty()) {
      names.emplace_back(name);
    }
  };

  append("image_encoder", parts.image_encoder);
  append("text_encoder", parts.text_encoder);
  append("geometry_encoder", parts.geometry_encoder);
  append("detector", parts.detector);
  return names;
}

void RunImagePredictor(const sam_s600::Sam3Manifest& manifest, const sam_s600::Sam3Request& request, bool require_all) {
  if (request.input_type != sam_s600::Sam3InputType::kImage || request.image_path.empty()) {
    throw std::runtime_error("--run for image mode requires --image raw-input-path");
  }

  CheckRequiredModelParts(manifest, require_all);
  sam_s600::Sam3ImagePredictor predictor(sam_s600::Sam3Model(manifest.config));
  const auto result = predictor.Predict(ReadRawImageBytes(request.image_path,
                                                         manifest.config.image_width,
                                                         manifest.config.image_height),
                                        request.prompt);
  PrintImageResult(result);
}

void RunImageInteractiveServer(const sam_s600::Sam3Manifest& manifest, bool require_all) {
  CheckRequiredModelParts(manifest, require_all);
  sam_s600::Sam3Model model(manifest.config);
  {
    StdoutToStderrGuard redirect_runtime_logs;
    model.LoadParts(ImageServeResidentParts(manifest), true);
  }
  sam_s600::Sam3ImagePredictor predictor(std::move(model));

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      break;
    }
    try {
      const auto request = ParseJsonLineRequest(line);
      const auto image = ReadRawImageBytes(request.image_path,
                                          manifest.config.image_width,
                                          manifest.config.image_height);
      const auto start = std::chrono::steady_clock::now();
      sam_s600::Sam3ImageResult result;
      {
        StdoutToStderrGuard redirect_runtime_logs;
        result = predictor.Predict(image, request.prompt);
      }
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      PrintJsonImageResult(result, elapsed_ms);
    } catch (const std::exception& error) {
      PrintJsonError(error.what());
    }
  }
}

void RunVideoPredictor(const sam_s600::Sam3Manifest& manifest, const sam_s600::Sam3Request& request, bool require_all) {
  if (request.input_type != sam_s600::Sam3InputType::kVideo || request.video_path.empty()) {
    throw std::runtime_error("--run for video mode requires --input raw-frame-path-or-directory");
  }

  sam_s600::Sam3Model model(manifest.config);
  model.Load(require_all);
  sam_s600::Sam3VideoPredictor predictor(std::move(model));
  const auto result = predictor.Predict(ReadRawVideoFrames(request.video_path,
                                                          manifest.config.image_width,
                                                          manifest.config.image_height),
                                        request.prompt);
  std::cout << "result_frames: " << result.frames.size() << '\n';
}

void RunMultiplexVideoPredictor(const sam_s600::Sam3Manifest& manifest, const sam_s600::Sam3Request& request, bool require_all) {
  if (request.input_type != sam_s600::Sam3InputType::kVideo || request.video_path.empty()) {
    throw std::runtime_error("--run for multiplex video mode requires --input raw-frame-path-or-directory");
  }

  sam_s600::Sam3Model model(manifest.config);
  model.Load(require_all);
  sam_s600::Sam3MultiplexVideoPredictor predictor(std::move(model));
  const auto result = predictor.Predict(ReadRawVideoFrames(request.video_path,
                                                          manifest.config.image_width,
                                                          manifest.config.image_height),
                                        request.prompt);
  std::cout << "result_frames: " << result.frames.size() << '\n';
}

}  // namespace

int RunManifestCli(int argc, char** argv, const std::string& default_manifest, const std::string& mode) {
  try {
    const auto options = ParseOptions(argc, argv, default_manifest);
    const auto manifest = sam_s600::LoadSam3Manifest(options.manifest_path);
    if (options.serve) {
      if (mode != "sam3_image_interactive") {
        throw std::runtime_error("--serve is currently implemented for sam3_image_interactive only");
      }
      RunImageInteractiveServer(manifest, options.require_all);
      return 0;
    }

    PrintManifest(manifest, mode);
    PrintRequest(options.request);
    if (options.check_models) {
      PrintModelChecks(manifest.config);
    }
    if (options.load_parts) {
      LoadModelParts(manifest, options.require_all);
    }
    if (options.run) {
      if (mode == "sam3_image" || mode == "sam3_image_interactive") {
        RunImagePredictor(manifest, options.request, options.require_all);
      } else if (mode == "sam3_video" || mode == "sam3_video_tracking") {
        RunVideoPredictor(manifest, options.request, options.require_all);
      } else if (mode == "sam3_multiplex_video") {
        RunMultiplexVideoPredictor(manifest, options.request, options.require_all);
      } else {
        throw std::runtime_error("--run is currently implemented for SAM3 image, video, and multiplex video modes only");
      }
    }
    if (!options.inspect_part_path.empty()) {
      InspectModelPart(options.inspect_part_path);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
