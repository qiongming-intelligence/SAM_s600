#include "sam_s600/sam3/sam3_image_predictor.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sam_s600/sam3/sam3_manifest.hpp"

namespace sam_s600 {
namespace {

bool HasPromptType(const Sam3Prompt& prompt, Sam3PromptType type) {
  return std::find(prompt.types.begin(), prompt.types.end(), type) != prompt.types.end();
}

void ValidatePrompt(const Sam3Prompt& prompt) {
  if (HasPromptType(prompt, Sam3PromptType::kText) && prompt.text.empty()) {
    throw std::runtime_error("text prompt is empty");
  }
  if (HasPromptType(prompt, Sam3PromptType::kPoint) && prompt.points.empty()) {
    throw std::runtime_error("point prompt is empty");
  }
  if (HasPromptType(prompt, Sam3PromptType::kBox) && prompt.boxes.empty()) {
    throw std::runtime_error("box prompt is empty");
  }
  if (HasPromptType(prompt, Sam3PromptType::kMask) && prompt.mask_path.empty()) {
    throw std::runtime_error("mask prompt path is empty");
  }
  if (HasPromptType(prompt, Sam3PromptType::kExemplar) && prompt.exemplar_path.empty()) {
    throw std::runtime_error("exemplar prompt path is empty");
  }
}

std::size_t TotalTensorBytes(const std::vector<BpuTensorBuffer>& buffers) {
  std::size_t total = 0;
  for (const auto& buffer : buffers) {
    total += buffer.buffer.Size();
  }
  return total;
}

bool SameShape(const TensorShape& lhs, const TensorShape& rhs) { return lhs.dims == rhs.dims; }

void ValidateCompatibleTensor(const BpuTensorBuffer& source, const BpuTensorBuffer& target, const std::string& stage_name) {
  if (source.info.dtype != target.info.dtype) {
    throw std::runtime_error(stage_name + " tensor dtype mismatch: " + target.info.name);
  }
  if (!SameShape(source.info.shape, target.info.shape)) {
    throw std::runtime_error(stage_name + " tensor shape mismatch: " + target.info.name);
  }
  if (source.buffer.Size() != target.buffer.Size()) {
    throw std::runtime_error(stage_name + " tensor byte size mismatch: " + target.info.name);
  }
}

void CopyBuffer(const BpuTensorBuffer& source, BpuTensorBuffer& target, const std::string& stage_name) {
  ValidateCompatibleTensor(source, target, stage_name);
  const auto bytes = source.buffer.Size();
  const auto* src = source.buffer.CpuData();
  auto* dst = target.buffer.CpuData();
  if (src == nullptr || dst == nullptr) {
    throw std::runtime_error(stage_name + " tensor binding has no CPU address: " + target.info.name);
  }
  std::copy_n(src, bytes, dst);
  target.buffer.CleanCache();
}

void FillTensorWithZeros(BpuTensorBuffer& tensor, const std::string& stage_name) {
  auto* dst = tensor.buffer.CpuData();
  if (dst == nullptr) {
    throw std::runtime_error(stage_name + " bridge tensor has no CPU address: " + tensor.info.name);
  }
  std::fill_n(dst, tensor.buffer.Size(), std::uint8_t{0});
  tensor.buffer.CleanCache();
}

void FillTensorWithOnes(BpuTensorBuffer& tensor, const std::string& stage_name) {
  auto* dst = tensor.buffer.CpuData();
  if (dst == nullptr) {
    throw std::runtime_error(stage_name + " bridge tensor has no CPU address: " + tensor.info.name);
  }
  if (tensor.info.dtype == TensorDataType::kInt32) {
    auto* values = reinterpret_cast<std::int32_t*>(dst);
    std::fill_n(values, tensor.buffer.Size() / sizeof(std::int32_t), std::int32_t{1});
  } else {
    std::fill_n(dst, tensor.buffer.Size(), std::uint8_t{1});
  }
  tensor.buffer.CleanCache();
}

// SAM3 uses -10 as the processor pad value for empty geometry prompts.
constexpr std::int32_t kSam3BoxLabelPadValue = -10;

void FillBoxLabelPadding(BpuTensorBuffer& tensor, const std::string& stage_name) {
  if (tensor.info.dtype != TensorDataType::kInt32) {
    throw std::runtime_error(stage_name + " input_boxes_labels is not int32: " + tensor.info.name);
  }
  auto* values = reinterpret_cast<std::int32_t*>(tensor.buffer.CpuData());
  if (values == nullptr) {
    throw std::runtime_error(stage_name + " bridge tensor has no CPU address: " + tensor.info.name);
  }
  std::fill_n(values, tensor.buffer.Size() / sizeof(std::int32_t), kSam3BoxLabelPadValue);
  tensor.buffer.CleanCache();
}

bool IsSam3MaskBridgeTensor(const std::string& name) {
  return name == "attention_mask" || name == "prompt_mask" || name == "/wrapped/geometry_encoder/Cast_6_output_0";
}

bool IsMaskDecoderStage(const std::string& stage_name) {
  return stage_name == "mask_decoder" || stage_name == "mask_decoder_pre_norm" ||
         stage_name == "mask_decoder_post_norm" || stage_name == "mask_decoder_pixel_mid" ||
         stage_name == "mask_decoder_pixel2_post_norm";
}

bool CanSynthesizeSam3BridgeTensor(const std::string& stage_name, const std::string& name) {
  if (stage_name == "detector_taps" || stage_name == "detector_bridge_taps" ||
      stage_name == "detector_image_bridge_taps" || stage_name == "detector_text_bridge_tap" ||
      stage_name == "detector_geometry_bridge_taps" || stage_name == "detector_encoder_hidden_tap") {
    return name == "input_ids" || name == "attention_mask" || name == "input_boxes" ||
           name == "input_boxes_labels" || name == "geometry_roi_features";
  }
  if (stage_name == "detector") {
    return name == "attention_mask" || name == "input_boxes" || name == "input_boxes_labels" ||
           name == "geometry_roi_features" || name == "/wrapped/geometry_encoder/Cast_6_output_0" ||
           name == "/wrapped/geometry_encoder/Transpose_1_output_0" ||
           name == "/wrapped/geometry_encoder/Transpose_output_0" ||
           name == "/wrapped/geometry_encoder/output_layer_norm/LayerNormalization_output_0" ||
           name == "/wrapped/text_projection/MatMul_output_0";
  }
  if (IsMaskDecoderStage(stage_name)) {
    return name == "decoder_queries" || name == "backbone_feature_0" || name == "backbone_feature_1" ||
           name == "encoder_hidden_states" || name == "prompt_features" || name == "prompt_mask" ||
           name == "/wrapped/Add_1_output_0" ||
           name == "/wrapped/pixel_decoder/norms.1/InstanceNormalization_output_0";
  }
  return false;
}

bool SynthesizeSam3BridgeTensor(const std::string& stage_name, BpuTensorBuffer& tensor) {
  if (!CanSynthesizeSam3BridgeTensor(stage_name, tensor.info.name)) {
    return false;
  }
  if (tensor.info.name == "input_boxes_labels") {
    FillBoxLabelPadding(tensor, stage_name);
  } else if (IsSam3MaskBridgeTensor(tensor.info.name)) {
    FillTensorWithOnes(tensor, stage_name);
  } else {
    FillTensorWithZeros(tensor, stage_name);
  }
  return true;
}

void CopyRawBytesToTensors(const std::vector<std::uint8_t>& bytes,
                           const std::string& stage_name,
                           std::vector<BpuTensorBuffer>& inputs) {
  if (bytes.empty()) {
    throw std::runtime_error(stage_name + " input bytes are empty");
  }
  if (inputs.empty()) {
    throw std::runtime_error(stage_name + " has no inputs");
  }

  const std::size_t required_bytes = TotalTensorBytes(inputs);
  if (bytes.size() > required_bytes) {
    throw std::runtime_error(stage_name + " input bytes exceed tensor inputs");
  }

  std::size_t offset = 0;
  for (auto& input : inputs) {
    auto* dst = input.buffer.CpuData();
    if (dst == nullptr) {
      throw std::runtime_error(stage_name + " input buffer has no CPU address");
    }
    const auto remaining = bytes.size() - offset;
    const auto copy_bytes = std::min<std::size_t>(remaining, input.buffer.Size());
    if (copy_bytes > 0) {
      std::copy_n(bytes.data() + offset, copy_bytes, dst);
    }
    if (copy_bytes < input.buffer.Size()) {
      std::fill_n(dst + copy_bytes, input.buffer.Size() - copy_bytes, std::uint8_t{0});
    }
    input.buffer.CleanCache();
    offset += copy_bytes;
  }
}

void CopyImageBytesToTensor(const Image& image, const std::string& stage_name, BpuTensorBuffer& input) {
  if (image.data.empty()) {
    throw std::runtime_error("image data is empty");
  }
  if (image.data.size() != input.buffer.Size()) {
    throw std::runtime_error("image data size does not match " + stage_name + " input: " + input.info.name);
  }
  auto* dst = input.buffer.CpuData();
  if (dst == nullptr) {
    throw std::runtime_error(stage_name + " input buffer has no CPU address: " + input.info.name);
  }
  std::copy_n(image.data.data(), image.data.size(), dst);
  input.buffer.CleanCache();
}

void CopyImageBytesToTensors(const Image& image, std::vector<BpuTensorBuffer>& inputs) {
  if (image.data.empty()) {
    throw std::runtime_error("image data is empty");
  }
  if (image.data.size() != TotalTensorBytes(inputs)) {
    throw std::runtime_error("image data size does not match SAM3 image encoder inputs");
  }
  CopyRawBytesToTensors(image.data, "image encoder", inputs);
}

void AppendBytes(std::vector<std::uint8_t>& bytes, const void* src, std::size_t size) {
  const auto* first = static_cast<const std::uint8_t*>(src);
  bytes.insert(bytes.end(), first, first + size);
}

void AppendFloat(std::vector<std::uint8_t>& bytes, float value) { AppendBytes(bytes, &value, sizeof(value)); }

void AppendInt(std::vector<std::uint8_t>& bytes, int value) { AppendBytes(bytes, &value, sizeof(value)); }

std::string Utf8FromCodepoint(int codepoint) {
  std::string out;
  if (codepoint <= 0x7F) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
  return out;
}

std::string ParseJsonString(const std::string& text, std::size_t& pos) {
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
        if (pos + 4 > text.size()) {
          throw std::runtime_error("invalid JSON unicode escape");
        }
        int codepoint = 0;
        for (int i = 0; i < 4; ++i) {
          const char hex = text[pos++];
          codepoint <<= 4;
          if (hex >= '0' && hex <= '9') {
            codepoint += hex - '0';
          } else if (hex >= 'a' && hex <= 'f') {
            codepoint += 10 + hex - 'a';
          } else if (hex >= 'A' && hex <= 'F') {
            codepoint += 10 + hex - 'A';
          } else {
            throw std::runtime_error("invalid JSON unicode escape");
          }
        }
        out += Utf8FromCodepoint(codepoint);
        break;
      }
      default:
        throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

std::unordered_map<std::string, int> LoadVocab(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open SAM3 tokenizer vocab: " + path.string());
  }
  const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  std::unordered_map<std::string, int> vocab;
  std::size_t pos = 0;
  while (pos < json.size()) {
    if (json[pos] != '"') {
      ++pos;
      continue;
    }
    auto key = ParseJsonString(json, pos);
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) {
      ++pos;
    }
    if (pos >= json.size() || json[pos] != ':') {
      continue;
    }
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) {
      ++pos;
    }
    int value = 0;
    bool has_digit = false;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
      has_digit = true;
      value = value * 10 + (json[pos++] - '0');
    }
    if (has_digit) {
      vocab.emplace(std::move(key), value);
    }
  }
  return vocab;
}

std::unordered_map<std::string, int> LoadMergeRanks(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open SAM3 tokenizer merges: " + path.string());
  }
  std::unordered_map<std::string, int> ranks;
  std::string line;
  int rank = 0;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream stream(line);
    std::string first;
    std::string second;
    if (stream >> first >> second) {
      ranks.emplace(first + '\n' + second, rank++);
    }
  }
  return ranks;
}

const std::vector<std::string>& ClipByteEncoder() {
  static const std::vector<std::string> encoder = [] {
    std::vector<int> bytes;
    for (int value = static_cast<int>('!'); value <= static_cast<int>('~'); ++value) bytes.push_back(value);
    for (int value = 0xA1; value <= 0xAC; ++value) bytes.push_back(value);
    for (int value = 0xAE; value <= 0xFF; ++value) bytes.push_back(value);

    std::vector<std::string> result(256);
    int extra = 0;
    for (int value : bytes) {
      result[static_cast<std::size_t>(value)] = Utf8FromCodepoint(value);
    }
    for (int value = 0; value < 256; ++value) {
      if (result[static_cast<std::size_t>(value)].empty()) {
        result[static_cast<std::size_t>(value)] = Utf8FromCodepoint(256 + extra++);
      }
    }
    return result;
  }();
  return encoder;
}

class ClipTokenizer {
 public:
  explicit ClipTokenizer(std::filesystem::path model_dir)
      : vocab_(LoadVocab(model_dir / "vocab.json")), merge_ranks_(LoadMergeRanks(model_dir / "merges.txt")) {}

  std::vector<int> Encode(const std::string& text, int max_tokens) const {
    std::vector<int> ids;
    ids.push_back(kBosToken);
    for (const auto& token : PreTokenize(text)) {
      for (const auto& piece : Bpe(token)) {
        const auto found = vocab_.find(piece);
        ids.push_back(found == vocab_.end() ? kEosToken : found->second);
      }
    }
    ids.push_back(kEosToken);
    if (max_tokens > 0 && static_cast<int>(ids.size()) > max_tokens) {
      ids.resize(static_cast<std::size_t>(max_tokens));
      ids.back() = kEosToken;
    }
    return ids;
  }

 private:
  static constexpr int kBosToken = 49406;
  static constexpr int kEosToken = 49407;

  static std::vector<std::string> PreTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    enum class Kind { kNone, kWord, kNumber, kPunctuation };
    Kind current_kind = Kind::kNone;
    const auto flush = [&tokens, &current] {
      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }
    };

    for (unsigned char ch : text) {
      if (std::isspace(ch) != 0) {
        flush();
        current_kind = Kind::kNone;
        continue;
      }
      Kind kind = Kind::kPunctuation;
      if (std::isalpha(ch) != 0) {
        kind = Kind::kWord;
        ch = static_cast<unsigned char>(std::tolower(ch));
      } else if (std::isdigit(ch) != 0) {
        kind = Kind::kNumber;
      }
      if (kind != current_kind && current_kind != Kind::kNone) {
        flush();
      }
      current_kind = kind;
      current.push_back(static_cast<char>(ch));
    }
    flush();
    return tokens;
  }

  std::vector<std::string> Bpe(const std::string& token) const {
    const auto& byte_encoder = ClipByteEncoder();
    std::vector<std::string> word;
    word.reserve(token.size());
    for (unsigned char ch : token) {
      word.push_back(byte_encoder[static_cast<std::size_t>(ch)]);
    }
    if (word.empty()) {
      return word;
    }
    word.back() += "</w>";

    while (word.size() > 1) {
      int best_rank = std::numeric_limits<int>::max();
      std::size_t best_index = word.size();
      for (std::size_t i = 0; i + 1 < word.size(); ++i) {
        const auto found = merge_ranks_.find(word[i] + '\n' + word[i + 1]);
        if (found != merge_ranks_.end() && found->second < best_rank) {
          best_rank = found->second;
          best_index = i;
        }
      }
      if (best_index == word.size()) {
        break;
      }
      std::vector<std::string> merged;
      merged.reserve(word.size() - 1);
      for (std::size_t i = 0; i < word.size(); ++i) {
        if (i == best_index) {
          merged.push_back(word[i] + word[i + 1]);
          ++i;
        } else {
          merged.push_back(word[i]);
        }
      }
      word = std::move(merged);
    }
    return word;
  }

  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> merge_ranks_;
};

const ClipTokenizer& Sam3ClipTokenizer() {
  static const ClipTokenizer tokenizer(std::filesystem::path{"models/upstream/modelscope/sam3"});
  return tokenizer;
}

int TensorElementCount(const TensorInfo& info) {
  int count = 1;
  for (int dim : info.shape.dims) {
    count *= dim;
  }
  return count;
}

void FillTextPromptInputs(const Sam3Prompt& prompt, std::vector<BpuTensorBuffer>& inputs, const std::string& stage_name) {
  for (auto& input : inputs) {
    if (input.info.name != "input_ids" && input.info.name != "attention_mask") {
      continue;
    }
    if (input.info.dtype != TensorDataType::kInt32) {
      throw std::runtime_error(stage_name + " text input is not int32: " + input.info.name);
    }
    auto* values = reinterpret_cast<std::int32_t*>(input.buffer.CpuData());
    if (values == nullptr) {
      throw std::runtime_error(stage_name + " text input has no CPU address: " + input.info.name);
    }
    const int value_count = static_cast<int>(input.buffer.Size() / sizeof(std::int32_t));
    std::fill_n(values, value_count, std::int32_t{0});

    const int token_count = TensorElementCount(input.info);
    const auto ids = Sam3ClipTokenizer().Encode(prompt.text, token_count);
    if (input.info.name == "input_ids") {
      std::fill_n(values, token_count, std::int32_t{49407});
      for (std::size_t i = 0; i < ids.size() && i < static_cast<std::size_t>(token_count); ++i) {
        values[i] = ids[i];
      }
    } else {
      for (std::size_t i = 0; i < ids.size() && i < static_cast<std::size_t>(token_count); ++i) {
        values[i] = 1;
      }
    }
    input.buffer.CleanCache();
  }
}

std::vector<BpuTensorBuffer> AllocateTextPromptInputs(const BpuModel& model,
                                                       const Sam3Prompt& prompt,
                                                       const std::string& stage_name) {
  auto inputs = model.AllocateInputs();
  FillTextPromptInputs(prompt, inputs, stage_name);
  return inputs;
}

std::vector<std::uint8_t> PackGeometryPromptBytes(const Sam3Prompt& prompt) {
  std::vector<std::uint8_t> bytes;
  AppendInt(bytes, static_cast<int>(prompt.points.size()));
  for (const auto& point : prompt.points) {
    AppendFloat(bytes, point.x);
    AppendFloat(bytes, point.y);
    AppendInt(bytes, point.label);
  }
  AppendInt(bytes, static_cast<int>(prompt.boxes.size()));
  for (const auto& box : prompt.boxes) {
    AppendFloat(bytes, box.x0);
    AppendFloat(bytes, box.y0);
    AppendFloat(bytes, box.x1);
    AppendFloat(bytes, box.y1);
  }
  AppendInt(bytes, prompt.mask_path.empty() ? 0 : 1);
  AppendInt(bytes, prompt.exemplar_path.empty() ? 0 : 1);
  return bytes;
}

std::vector<BpuTensorBuffer> RunPromptEncoder(const BpuModel& encoder,
                                              const std::string& stage_name,
                                              const std::vector<std::uint8_t>& bytes) {
  auto inputs = encoder.AllocateInputs();
  auto outputs = encoder.AllocateOutputs();
  CopyRawBytesToTensors(bytes, stage_name, inputs);
  encoder.Infer(inputs, outputs);
  return outputs;
}

std::vector<BpuTensorBuffer> RunTextEncoder(const BpuModel& encoder, const Sam3Prompt& prompt) {
  auto inputs = AllocateTextPromptInputs(encoder, prompt, "text_encoder");
  auto outputs = encoder.AllocateOutputs();
  encoder.Infer(inputs, outputs);
  return outputs;
}

const BpuTensorBuffer* FindTensor(const std::vector<const BpuTensorBuffer*>& sources, const std::string& name) {
  for (const auto* source : sources) {
    if (source != nullptr && source->info.name == name) {
      return source;
    }
  }
  return nullptr;
}

const BpuTensorBuffer* FindSourceTensor(const std::vector<const BpuTensorBuffer*>& sources, const std::string& name) {
  if (const auto* exact = FindTensor(sources, name)) {
    return exact;
  }
  if (name == "pixel_values") {
    return FindTensor(sources, "image");
  }
  if (name == "encoder_hidden_states") {
    return FindTensor(sources, "/wrapped/detr_encoder/layers.5/Add_3_output_0");
  }
  return nullptr;
}

std::vector<const BpuTensorBuffer*> TensorRefs(const std::vector<BpuTensorBuffer>& buffers) {
  std::vector<const BpuTensorBuffer*> refs;
  refs.reserve(buffers.size());
  for (const auto& buffer : buffers) {
    refs.push_back(&buffer);
  }
  return refs;
}

void AppendTensorRefs(std::vector<const BpuTensorBuffer*>& refs, const std::vector<BpuTensorBuffer>& buffers) {
  refs.reserve(refs.size() + buffers.size());
  for (const auto& buffer : buffers) {
    refs.push_back(&buffer);
  }
}

bool UsesTextPromptInputs(const std::string& stage_name) {
  return stage_name == "detector" || stage_name == "detector_taps" || stage_name == "detector_bridge_taps" ||
         stage_name == "detector_text_bridge_tap" || stage_name == "detector_encoder_hidden_tap";
}

BpuTensorBuffer BuildVisionPositionBridgeTensor(const TensorInfo& info);

bool DebugBindingsEnabled() {
  const char* value = std::getenv("SAM3_DEBUG_BIND");
  return value != nullptr && std::string_view{value} != "0";
}

std::string SafeTensorFileName(std::string name) {
  for (char& ch : name) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '.' || ch == '_' || ch == '-')) {
      ch = '_';
    }
  }
  return name;
}

void DumpDebugTensor(const std::string& stage_name, const BpuTensorBuffer& tensor) {
  const char* dir = std::getenv("SAM3_DUMP_TENSORS_DIR");
  if (dir == nullptr || *dir == '\0') {
    return;
  }
  std::filesystem::create_directories(dir);
  const auto path = std::filesystem::path{dir} / (stage_name + "__" + SafeTensorFileName(tensor.info.name) + ".bin");
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to open SAM3 tensor dump: " + path.string());
  }
  const auto* data = tensor.buffer.CpuData();
  if (data == nullptr) {
    throw std::runtime_error("SAM3 tensor dump has no CPU address: " + tensor.info.name);
  }
  output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(tensor.buffer.Size()));
}

bool LoadDebugTensor(const std::string& stage_name, BpuTensorBuffer& tensor) {
  const char* dir = std::getenv("SAM3_LOAD_TENSORS_DIR");
  if (dir == nullptr || *dir == '\0') {
    return false;
  }
  const auto path = std::filesystem::path{dir} / (stage_name + "__" + SafeTensorFileName(tensor.info.name) + ".bin");
  if (!std::filesystem::exists(path)) {
    return false;
  }
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("failed to open SAM3 tensor override: " + path.string());
  }
  const auto size = input.tellg();
  if (size < 0 || static_cast<std::uint64_t>(size) != tensor.buffer.Size()) {
    throw std::runtime_error("SAM3 tensor override size mismatch for " + tensor.info.name + ": " + path.string());
  }
  input.seekg(0, std::ios::beg);
  auto* data = tensor.buffer.CpuData();
  if (data == nullptr) {
    throw std::runtime_error("SAM3 tensor override has no CPU address: " + tensor.info.name);
  }
  input.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(tensor.buffer.Size()));
  if (!input) {
    throw std::runtime_error("failed to read SAM3 tensor override: " + path.string());
  }
  tensor.buffer.CleanCache();
  return true;
}

void PrintDebugTensorSummary(const std::string& stage_name, const BpuTensorBuffer& tensor, const char* origin) {
  if (!DebugBindingsEnabled()) {
    return;
  }
  std::cerr << "[sam3-debug] " << stage_name << " " << origin << " " << tensor.info.name << " shape=(";
  for (std::size_t i = 0; i < tensor.info.shape.dims.size(); ++i) {
    if (i != 0) {
      std::cerr << ',';
    }
    std::cerr << tensor.info.shape.dims[i];
  }
  std::cerr << ") bytes=" << tensor.buffer.Size();
  if (tensor.info.dtype == TensorDataType::kFloat32 && tensor.buffer.Size() >= sizeof(float)) {
    const auto* values = reinterpret_cast<const float*>(tensor.buffer.CpuData());
    const std::size_t count = tensor.buffer.Size() / sizeof(float);
    const std::size_t sample_count = std::min<std::size_t>(count, 1024);
    double sum = 0.0;
    float min_value = values[0];
    float max_value = values[0];
    for (std::size_t i = 0; i < sample_count; ++i) {
      min_value = std::min(min_value, values[i]);
      max_value = std::max(max_value, values[i]);
      sum += values[i];
    }
    std::cerr << " sample_f32_min=" << min_value << " max=" << max_value << " mean="
              << (sum / static_cast<double>(sample_count));
    std::cerr << " first=" << values[0];
  } else if (tensor.info.dtype == TensorDataType::kInt32 && tensor.buffer.Size() >= sizeof(std::int32_t)) {
    const auto* values = reinterpret_cast<const std::int32_t*>(tensor.buffer.CpuData());
    const std::size_t count = std::min<std::size_t>(tensor.buffer.Size() / sizeof(std::int32_t), 8);
    std::cerr << " first_s32=";
    for (std::size_t i = 0; i < count; ++i) {
      if (i != 0) {
        std::cerr << ',';
      }
      std::cerr << values[i];
    }
  } else if (tensor.info.dtype == TensorDataType::kBool8 && tensor.buffer.Size() > 0) {
    const auto* values = tensor.buffer.CpuData();
    const std::size_t count = std::min<std::size_t>(tensor.buffer.Size(), 8);
    std::cerr << " first_bool8=";
    for (std::size_t i = 0; i < count; ++i) {
      if (i != 0) {
        std::cerr << ',';
      }
      std::cerr << static_cast<int>(values[i]);
    }
  }
  std::cerr << '\n';
}

void BindInputsByName(const std::string& stage_name,
                      const std::vector<const BpuTensorBuffer*>& sources,
                      std::vector<BpuTensorBuffer>& inputs,
                      const Sam3Prompt* prompt = nullptr,
                      const Image* image = nullptr) {
  const bool fill_text_inputs = prompt != nullptr && UsesTextPromptInputs(stage_name);
  if (fill_text_inputs) {
    FillTextPromptInputs(*prompt, inputs, stage_name);
    for (const auto& input : inputs) {
      if (input.info.name == "input_ids" || input.info.name == "attention_mask") {
        PrintDebugTensorSummary(stage_name, input, "text");
      }
    }
  }
  for (auto& input : inputs) {
    if (fill_text_inputs && (input.info.name == "input_ids" || input.info.name == "attention_mask")) {
      continue;
    }
    if (image != nullptr && (input.info.name == "pixel_values" || input.info.name == "image")) {
      CopyImageBytesToTensor(*image, stage_name, input);
      PrintDebugTensorSummary(stage_name, input, "image");
      DumpDebugTensor(stage_name, input);
      continue;
    }
    if (LoadDebugTensor(stage_name, input)) {
      PrintDebugTensorSummary(stage_name, input, "override");
      continue;
    }
    const auto* source = FindSourceTensor(sources, input.info.name);
    if (source == nullptr) {
      if (stage_name == "detector" && input.info.name == "/wrapped/geometry_encoder/Transpose_1_output_0") {
        auto position_bridge = BuildVisionPositionBridgeTensor(input.info);
        CopyBuffer(position_bridge, input, stage_name);
        PrintDebugTensorSummary(stage_name, input, "generated_position");
        DumpDebugTensor(stage_name, input);
        continue;
      }
      if (SynthesizeSam3BridgeTensor(stage_name, input)) {
        PrintDebugTensorSummary(stage_name, input, "synthesized");
        DumpDebugTensor(stage_name, input);
        continue;
      }
      throw std::runtime_error(stage_name + " input tensor has no upstream binding: " + input.info.name);
    }
    CopyBuffer(*source, input, stage_name);
    PrintDebugTensorSummary(stage_name, input, "source");
    DumpDebugTensor(stage_name, input);
  }
}

std::vector<BpuTensorBuffer> RunStageFromSources(const BpuModel& stage,
                                                 const std::string& stage_name,
                                                 const std::vector<const BpuTensorBuffer*>& sources,
                                                 const Sam3Prompt* prompt = nullptr,
                                                 const Image* image = nullptr) {
  auto inputs = stage.AllocateInputs();
  auto outputs = stage.AllocateOutputs();
  BindInputsByName(stage_name, sources, inputs, prompt, image);
  stage.Infer(inputs, outputs);
  return outputs;
}

BpuTensorBuffer BuildVisionPositionBridgeTensor(const TensorInfo& info) {
  if (info.name != "/wrapped/geometry_encoder/Transpose_1_output_0" || info.dtype != TensorDataType::kFloat32 ||
      info.shape.dims != std::vector<int>{1, 5184, 256}) {
    throw std::runtime_error("unexpected SAM3 position bridge tensor info: " + info.name);
  }

  BpuAllocator allocator;
  auto output = allocator.AllocateTensor(info);
  auto* values = reinterpret_cast<float*>(output.buffer.CpuData());
  if (values == nullptr) {
    throw std::runtime_error("SAM3 position bridge tensor has no CPU address");
  }

  constexpr int kHeight = 72;
  constexpr int kWidth = 72;
  constexpr int kFeatures = 128;
  constexpr float kScale = 6.28318530717958647692F;
  constexpr float kEps = 1.0e-6F;
  constexpr float kTemperature = 10000.0F;

  float dim_t[kFeatures];
  for (int feature = 0; feature < kFeatures; ++feature) {
    dim_t[feature] = std::pow(kTemperature, 2.0F * static_cast<float>(feature / 2) / static_cast<float>(kFeatures));
  }

  for (int y = 0; y < kHeight; ++y) {
    const float y_embed = (static_cast<float>(y + 1) / (static_cast<float>(kHeight) + kEps)) * kScale;
    for (int x = 0; x < kWidth; ++x) {
      const float x_embed = (static_cast<float>(x + 1) / (static_cast<float>(kWidth) + kEps)) * kScale;
      auto* row = values + (y * kWidth + x) * 256;
      for (int feature = 0; feature < kFeatures / 2; ++feature) {
        const int even = feature * 2;
        row[even] = std::sin(y_embed / dim_t[even]);
        row[even + 1] = std::cos(y_embed / dim_t[even + 1]);
        row[128 + even] = std::sin(x_embed / dim_t[even]);
        row[128 + even + 1] = std::cos(x_embed / dim_t[even + 1]);
      }
    }
  }

  output.buffer.CleanCache();
  return output;
}

BpuTensorBuffer CpuHighResolutionInstanceNorm(const BpuTensorBuffer& input, const TensorInfo& output_info) {
  if (input.info.dtype != TensorDataType::kFloat32 || output_info.dtype != TensorDataType::kFloat32) {
    throw std::runtime_error("mask decoder CPU norm expects float32 tensors");
  }
  if (input.info.shape.dims != std::vector<int>{1, 256, 252, 252}) {
    throw std::runtime_error("mask decoder CPU norm input shape mismatch: " + input.info.name);
  }
  if (output_info.shape.dims != std::vector<int>{1, 8, 2032128}) {
    throw std::runtime_error("mask decoder CPU norm output shape mismatch: " + output_info.name);
  }

  BpuAllocator allocator;
  auto output = allocator.AllocateTensor(output_info);
  const auto* src = reinterpret_cast<const float*>(input.buffer.CpuData());
  auto* dst = reinterpret_cast<float*>(output.buffer.CpuData());
  if (src == nullptr || dst == nullptr) {
    throw std::runtime_error("mask decoder CPU norm tensor has no CPU address");
  }

  constexpr int kGroups = 8;
  constexpr int kChannels = 256;
  constexpr int kHeight = 252;
  constexpr int kWidth = 252;
  constexpr int kChannelsPerGroup = kChannels / kGroups;
  constexpr int kSpatial = kHeight * kWidth;
  constexpr int kGroupSize = kChannelsPerGroup * kSpatial;
  constexpr float kEpsilon = 9.999999747378752e-06F;

  for (int group = 0; group < kGroups; ++group) {
    const int channel_offset = group * kChannelsPerGroup;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int channel = 0; channel < kChannelsPerGroup; ++channel) {
      const auto* channel_src = src + (channel_offset + channel) * kSpatial;
      for (int index = 0; index < kSpatial; ++index) {
        const double value = channel_src[index];
        sum += value;
        sum_sq += value * value;
      }
    }
    const double mean = sum / kGroupSize;
    const double variance = std::max(0.0, sum_sq / kGroupSize - mean * mean);
    const float scale = 1.0F / std::sqrt(static_cast<float>(variance) + kEpsilon);
    auto* group_dst = dst + group * kGroupSize;
    for (int channel = 0; channel < kChannelsPerGroup; ++channel) {
      const auto* channel_src = src + (channel_offset + channel) * kSpatial;
      for (int index = 0; index < kSpatial; ++index) {
        group_dst[channel * kSpatial + index] = (channel_src[index] - static_cast<float>(mean)) * scale;
      }
    }
  }

  output.buffer.CleanCache();
  return output;
}

std::string ConfiguredPartPath(const Sam3Config& config, const std::string& name) {
  for (const auto& part : ListSam3ModelParts(config)) {
    if (part.name == name) {
      return part.path;
    }
  }
  return {};
}

bool HasConfiguredPart(const Sam3Config& config, const std::string& name) {
  return !ConfiguredPartPath(config, name).empty();
}

void RequireConfiguredPart(const Sam3Model& model, const std::string& name) {
  if (!HasConfiguredPart(model.Config(), name)) {
    throw std::runtime_error("missing required SAM3 model part: " + name);
  }
}

template <typename Callback>
auto WithConfiguredStage(const Sam3Model& model, const char* name, Callback callback)
    -> decltype(callback(std::declval<const BpuModel&>())) {
  if (const BpuModel* part = model.FindPart(name)) {
    if (!part->Loaded()) {
      throw std::runtime_error("SAM3 model part is not loaded: " + std::string{name});
    }
    return callback(*part);
  }

  const auto path = ConfiguredPartPath(model.Config(), name);
  if (path.empty()) {
    throw std::runtime_error("missing required SAM3 model part: " + std::string{name});
  }
  BpuModel stage(path);
  return callback(stage);
}

std::vector<BpuTensorBuffer> RunConfiguredStageFromSources(const Sam3Model& model,
                                                           const char* stage_name,
                                                           const std::vector<const BpuTensorBuffer*>& sources,
                                                           const Sam3Prompt* prompt = nullptr,
                                                           const Image* image = nullptr) {
  return WithConfiguredStage(model, stage_name, [&](const BpuModel& stage) {
    return RunStageFromSources(stage, stage_name, sources, prompt, image);
  });
}

bool HasRealSplitMaskDecoder(const Sam3Model& model) {
  return HasConfiguredPart(model.Config(), "mask_decoder_pre_norm") &&
         HasConfiguredPart(model.Config(), "mask_decoder_post_norm");
}

bool HasSmokeSplitMaskDecoder(const Sam3Model& model) {
  return HasConfiguredPart(model.Config(), "mask_decoder_pixel_mid") &&
         HasConfiguredPart(model.Config(), "mask_decoder_pixel2_post_norm");
}

bool HasTensor(const std::vector<BpuTensorBuffer>& tensors, const std::string& name) {
  return std::any_of(tensors.begin(), tensors.end(), [&](const BpuTensorBuffer& tensor) {
    return tensor.info.name == name;
  });
}

bool DetectorHasMaskOutputs(const std::vector<BpuTensorBuffer>& detector_outputs) {
  return HasTensor(detector_outputs, "mask_logits") && HasTensor(detector_outputs, "semantic_segmentation");
}

bool DetectorAcceptsImageInput(const Sam3Model& model) {
  return WithConfiguredStage(model, "detector", [](const BpuModel& detector) {
    return std::any_of(detector.Inputs().begin(), detector.Inputs().end(), [](const TensorInfo& input) {
      return input.name == "pixel_values" || input.name == "image";
    });
  });
}

void RequirePromptParts(const Sam3Model& model, const Sam3Prompt& prompt) {
  RequireConfiguredPart(model, "detector");
  if (DetectorAcceptsImageInput(model)) {
    return;
  }
  RequireConfiguredPart(model, "image_encoder");

  if (HasPromptType(prompt, Sam3PromptType::kText)) {
    RequireConfiguredPart(model, "text_encoder");
  }
  if (HasPromptType(prompt, Sam3PromptType::kPoint) || HasPromptType(prompt, Sam3PromptType::kBox) ||
      HasPromptType(prompt, Sam3PromptType::kMask) || HasPromptType(prompt, Sam3PromptType::kExemplar)) {
    RequireConfiguredPart(model, "geometry_encoder");
  }
}

const BpuTensorBuffer* FindOptionalTensor(const std::vector<BpuTensorBuffer>& buffers, std::string_view name) {
  for (const auto& buffer : buffers) {
    if (buffer.info.name == name) {
      return &buffer;
    }
  }
  return nullptr;
}

const BpuTensorBuffer& RequireTensor(const std::vector<BpuTensorBuffer>& buffers, std::string_view name) {
  if (const auto* tensor = FindOptionalTensor(buffers, name)) {
    return *tensor;
  }
  throw std::runtime_error("missing tensor output: " + std::string{name});
}

const BpuTensorBuffer& RequireTensorWithShape(const std::vector<BpuTensorBuffer>& buffers,
                                              const std::vector<int>& dims,
                                              std::string_view context) {
  for (const auto& buffer : buffers) {
    if (buffer.info.shape.dims == dims) {
      return buffer;
    }
  }
  throw std::runtime_error("missing tensor for " + std::string{context});
}

float Sigmoid(float value) {
  if (value >= 0.0F) {
    const float z = std::exp(-value);
    return 1.0F / (1.0F + z);
  }
  const float z = std::exp(value);
  return z / (1.0F + z);
}

int ElementCount(const TensorShape& shape) {
  if (shape.dims.empty()) {
    return 0;
  }
  int count = 1;
  for (const auto dim : shape.dims) {
    count *= dim;
  }
  return count;
}

float Float16ToFloat(std::uint16_t bits) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000U)) << 16U;
  std::uint32_t exponent = (bits >> 10U) & 0x1FU;
  std::uint32_t mantissa = bits & 0x03FFU;
  std::uint32_t value = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      value = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        --exponent;
      }
      mantissa &= 0x03FFU;
      value = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1FU) {
    value = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    value = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float result = 0.0F;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

class FloatTensorView {
 public:
  explicit FloatTensorView(const BpuTensorBuffer& tensor, std::string_view expected_name) {
    if (tensor.info.name != expected_name) {
      throw std::runtime_error("unexpected tensor name: " + tensor.info.name);
    }
    const auto* data = tensor.buffer.CpuData();
    if (data == nullptr) {
      throw std::runtime_error("tensor has no CPU address: " + tensor.info.name);
    }
    if (tensor.info.dtype == TensorDataType::kFloat32) {
      data_ = reinterpret_cast<const float*>(data);
      size_ = tensor.buffer.Size() / sizeof(float);
      return;
    }
    if (tensor.info.dtype == TensorDataType::kFloat16) {
      size_ = tensor.buffer.Size() / sizeof(std::uint16_t);
      owned_ = std::make_unique<float[]>(size_);
      const auto* values = reinterpret_cast<const std::uint16_t*>(data);
      for (std::size_t i = 0; i < size_; ++i) {
        owned_[i] = Float16ToFloat(values[i]);
      }
      data_ = owned_.get();
      return;
    }
    throw std::runtime_error("tensor is not float32/float16: " + tensor.info.name);
  }

  [[nodiscard]] const float* data() const { return data_; }
  [[nodiscard]] std::size_t size() const { return size_; }

 private:
  std::unique_ptr<float[]> owned_;
  const float* data_{nullptr};
  std::size_t size_{0};
};

void AttachMasksToObjects(std::vector<Sam3Object>& objects, const BpuTensorBuffer& mask_logits_tensor) {
  const FloatTensorView data_view(mask_logits_tensor, "mask_logits");
  const auto* data = data_view.data();
  const auto& dims = mask_logits_tensor.info.shape.dims;
  if (dims.size() != 4 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0) {
    throw std::runtime_error("mask_logits shape is invalid");
  }

  const int object_count = dims[1];
  const int height = dims[2];
  const int width = dims[3];
  const int mask_size = height * width;
  for (auto& object : objects) {
    if (object.id < 0 || object.id >= object_count) {
      continue;
    }
    const auto* begin = data + object.id * mask_size;
    object.mask.width = width;
    object.mask.height = height;
    object.mask.logits.assign(begin, begin + mask_size);
  }
}

std::vector<Sam3Object> BuildDetectorObjects(const BpuTensorBuffer& logits_tensor,
                                             const BpuTensorBuffer& boxes_tensor,
                                             float presence_score,
                                             float score_threshold,
                                             int limit) {
  const FloatTensorView logits_view(logits_tensor, "object_logits");
  const FloatTensorView boxes_view(boxes_tensor, "object_boxes");
  const auto* logits = logits_view.data();
  const auto* boxes = boxes_view.data();
  const int object_count = ElementCount(logits_tensor.info.shape);
  if (object_count <= 0 || ElementCount(boxes_tensor.info.shape) < object_count * 4) {
    throw std::runtime_error("detector output shape is invalid");
  }

  std::vector<Sam3Object> objects;
  objects.reserve(static_cast<std::size_t>(object_count));
  for (int index = 0; index < object_count; ++index) {
    const float score = Sigmoid(logits[index]);
    if (score < score_threshold) {
      continue;
    }
    Sam3Object object;
    object.id = index;
    object.score = score;
    object.presence_score = presence_score;
    object.x0 = boxes[index * 4 + 0];
    object.y0 = boxes[index * 4 + 1];
    object.x1 = boxes[index * 4 + 2];
    object.y1 = boxes[index * 4 + 3];
    objects.push_back(object);
  }

  std::sort(objects.begin(), objects.end(), [](const Sam3Object& lhs, const Sam3Object& rhs) {
    return lhs.score > rhs.score;
  });
  if (limit > 0 && static_cast<int>(objects.size()) > limit) {
    objects.resize(static_cast<std::size_t>(limit));
  }
  return objects;
}

Sam3ImageResult BuildImageResult(const std::vector<BpuTensorBuffer>& detector_outputs,
                                 const std::vector<BpuTensorBuffer>& mask_outputs) {
  const auto& logits_tensor = RequireTensor(detector_outputs, "object_logits");
  const auto& boxes_tensor = RequireTensor(detector_outputs, "object_boxes");
  const auto& presence_tensor = RequireTensor(detector_outputs, "presence_logits");
  const FloatTensorView presence_view(presence_tensor, "presence_logits");
  const auto* presence = presence_view.data();

  Sam3ImageResult result;
  result.presence_logit = presence[0];
  result.presence_score = Sigmoid(presence[0]);
  result.objects = BuildDetectorObjects(logits_tensor, boxes_tensor, result.presence_score, 0.5F, 0);
  result.candidates = BuildDetectorObjects(logits_tensor, boxes_tensor, result.presence_score, -std::numeric_limits<float>::infinity(), 20);
  const auto* mask_logits_tensor = FindOptionalTensor(mask_outputs, "mask_logits");
  if (mask_logits_tensor == nullptr) {
    mask_logits_tensor = FindOptionalTensor(detector_outputs, "mask_logits");
  }
  if (mask_logits_tensor != nullptr) {
    AttachMasksToObjects(result.objects, *mask_logits_tensor);
    AttachMasksToObjects(result.candidates, *mask_logits_tensor);
  }
  return result;
}

}  // namespace

Sam3ImagePredictor::Sam3ImagePredictor(Sam3Model model) : model_(std::move(model)) {}

Sam3ImageResult Sam3ImagePredictor::Predict(const Image& image, const Sam3Prompt& prompt) const {
  if (image.width <= 0 || image.height <= 0) {
    throw std::runtime_error("image dimensions are invalid");
  }

  ValidatePrompt(prompt);
  RequirePromptParts(model_, prompt);

  if (DetectorAcceptsImageInput(model_)) {
    const std::vector<const BpuTensorBuffer*> sources;
    auto detector_outputs = RunConfiguredStageFromSources(model_, "detector", sources, &prompt, &image);
    const std::vector<BpuTensorBuffer> mask_outputs;
    return BuildImageResult(detector_outputs, mask_outputs);
  }

  std::vector<BpuTensorBuffer> text_outputs;
  std::vector<BpuTensorBuffer> geometry_outputs;
  if (HasPromptType(prompt, Sam3PromptType::kText)) {
    text_outputs = WithConfiguredStage(model_, "text_encoder", [&](const BpuModel& text_encoder) {
      return RunTextEncoder(text_encoder, prompt);
    });
  }
  if (HasPromptType(prompt, Sam3PromptType::kPoint) || HasPromptType(prompt, Sam3PromptType::kBox) ||
      HasPromptType(prompt, Sam3PromptType::kMask) || HasPromptType(prompt, Sam3PromptType::kExemplar)) {
    geometry_outputs = WithConfiguredStage(model_, "geometry_encoder", [&](const BpuModel& geometry_encoder) {
      return RunPromptEncoder(geometry_encoder, "geometry_encoder", PackGeometryPromptBytes(prompt));
    });
  }

  std::vector<BpuTensorBuffer> image_inputs;
  std::vector<BpuTensorBuffer> image_outputs;
  WithConfiguredStage(model_, "image_encoder", [&](const BpuModel& image_encoder) {
    image_inputs = image_encoder.AllocateInputs();
    image_outputs = image_encoder.AllocateOutputs();
    CopyImageBytesToTensors(image, image_inputs);
    image_encoder.Infer(image_inputs, image_outputs);
  });

  std::vector<BpuTensorBuffer> detector_tap_outputs;
  auto tap_sources = TensorRefs(image_inputs);
  AppendTensorRefs(tap_sources, image_outputs);
  AppendTensorRefs(tap_sources, text_outputs);
  AppendTensorRefs(tap_sources, geometry_outputs);
  const auto& parts = model_.Config().model_parts;
  if (!parts.detector_taps.empty()) {
    detector_tap_outputs = RunConfiguredStageFromSources(model_, "detector_taps", tap_sources, &prompt);
  } else {
    if (!parts.detector_bridge_taps.empty()) {
      detector_tap_outputs = RunConfiguredStageFromSources(model_, "detector_bridge_taps", tap_sources, &prompt);
    } else {
      const char* micro_tap_names[] = {
          "detector_image_bridge_taps",
          "detector_text_bridge_tap",
          "detector_geometry_bridge_taps",
      };
      for (const auto* micro_tap_name : micro_tap_names) {
        const auto* micro_tap_path = micro_tap_name == std::string{"detector_image_bridge_taps"}
                                      ? &parts.detector_image_bridge_taps
                                      : micro_tap_name == std::string{"detector_text_bridge_tap"}
                                            ? &parts.detector_text_bridge_tap
                                            : &parts.detector_geometry_bridge_taps;
        if (micro_tap_path->empty()) {
          continue;
        }
        if (std::string_view{micro_tap_name} == "detector_text_bridge_tap" &&
            !HasPromptType(prompt, Sam3PromptType::kText)) {
          continue;
        }
        const Sam3Prompt* tap_prompt = UsesTextPromptInputs(micro_tap_name) ? &prompt : nullptr;
        auto micro_tap_outputs = RunConfiguredStageFromSources(model_, micro_tap_name, tap_sources, tap_prompt);
        for (auto& output : micro_tap_outputs) {
          detector_tap_outputs.push_back(std::move(output));
        }
        tap_sources = TensorRefs(image_inputs);
        AppendTensorRefs(tap_sources, image_outputs);
        AppendTensorRefs(tap_sources, text_outputs);
        AppendTensorRefs(tap_sources, geometry_outputs);
        AppendTensorRefs(tap_sources, detector_tap_outputs);
      }
    }
    if (!parts.detector_encoder_hidden_tap.empty()) {
      tap_sources = TensorRefs(image_inputs);
      AppendTensorRefs(tap_sources, image_outputs);
      AppendTensorRefs(tap_sources, text_outputs);
      AppendTensorRefs(tap_sources, geometry_outputs);
      AppendTensorRefs(tap_sources, detector_tap_outputs);
      auto encoder_tap_outputs = RunConfiguredStageFromSources(model_,
                                                               "detector_encoder_hidden_tap",
                                                               tap_sources,
                                                               &prompt);
      for (auto& output : encoder_tap_outputs) {
        detector_tap_outputs.push_back(std::move(output));
      }
    }
  }

  auto detector_sources = TensorRefs(image_inputs);
  AppendTensorRefs(detector_sources, image_outputs);
  AppendTensorRefs(detector_sources, text_outputs);
  AppendTensorRefs(detector_sources, geometry_outputs);
  AppendTensorRefs(detector_sources, detector_tap_outputs);
  auto detector_outputs = RunConfiguredStageFromSources(model_, "detector", detector_sources, &prompt);

  std::vector<BpuTensorBuffer> mask_outputs;
  if (DetectorHasMaskOutputs(detector_outputs)) {
  } else if (HasRealSplitMaskDecoder(model_)) {
    auto mask_sources = TensorRefs(image_outputs);
    AppendTensorRefs(mask_sources, text_outputs);
    AppendTensorRefs(mask_sources, geometry_outputs);
    AppendTensorRefs(mask_sources, detector_tap_outputs);
    AppendTensorRefs(mask_sources, detector_outputs);
    auto pre_norm_outputs = RunConfiguredStageFromSources(model_, "mask_decoder_pre_norm", mask_sources);
    WithConfiguredStage(model_, "mask_decoder_post_norm", [&](const BpuModel& mask_decoder_post_norm) {
      auto post_norm_inputs = mask_decoder_post_norm.AllocateInputs();
      if (post_norm_inputs.size() != 1) {
        throw std::runtime_error("mask_decoder_post_norm expects one input");
      }
      const auto& pre_norm_tensor = RequireTensorWithShape(pre_norm_outputs,
                                                           post_norm_inputs.front().info.shape.dims,
                                                           "mask_decoder_post_norm CPU norm");
      auto norm_output = CpuHighResolutionInstanceNorm(pre_norm_tensor, post_norm_inputs.front().info);
      std::vector<const BpuTensorBuffer*> post_norm_sources{&norm_output};
      auto post_norm_outputs = RunStageFromSources(mask_decoder_post_norm,
                                                   "mask_decoder_post_norm",
                                                   post_norm_sources);
      mask_outputs = std::move(pre_norm_outputs);
      for (auto& output : post_norm_outputs) {
        mask_outputs.push_back(std::move(output));
      }
    });
  } else if (HasSmokeSplitMaskDecoder(model_)) {
    auto mask_sources = TensorRefs(image_outputs);
    AppendTensorRefs(mask_sources, text_outputs);
    AppendTensorRefs(mask_sources, geometry_outputs);
    AppendTensorRefs(mask_sources, detector_tap_outputs);
    AppendTensorRefs(mask_sources, detector_outputs);
    auto pixel_mid_outputs = RunConfiguredStageFromSources(model_, "mask_decoder_pixel_mid", mask_sources);
    auto pixel2_sources = TensorRefs(pixel_mid_outputs);
    mask_outputs = RunConfiguredStageFromSources(model_, "mask_decoder_pixel2_post_norm", pixel2_sources);
  } else if (!parts.mask_decoder.empty()) {
    auto mask_sources = TensorRefs(image_outputs);
    AppendTensorRefs(mask_sources, text_outputs);
    AppendTensorRefs(mask_sources, geometry_outputs);
    AppendTensorRefs(mask_sources, detector_tap_outputs);
    AppendTensorRefs(mask_sources, detector_outputs);
    mask_outputs = RunConfiguredStageFromSources(model_, "mask_decoder", mask_sources);
  }

  return BuildImageResult(detector_outputs, mask_outputs);
}

}  // namespace sam_s600
