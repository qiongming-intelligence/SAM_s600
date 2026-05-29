#include "sam_s600/sam3/sam3_image_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sam_s600 {
namespace {

bool HasPromptType(const Sam3Prompt& prompt, Sam3PromptType type) {
  return std::find(prompt.types.begin(), prompt.types.end(), type) != prompt.types.end();
}

const BpuModel& RequirePart(const Sam3Model& model, const char* name) {
  const BpuModel* part = model.FindPart(name);
  if (part == nullptr) {
    throw std::runtime_error("missing required SAM3 model part: " + std::string{name});
  }
  if (!part->Loaded()) {
    throw std::runtime_error("SAM3 model part is not loaded: " + std::string{name});
  }
  return *part;
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
    return name == "attention_mask" || name == "/wrapped/geometry_encoder/Cast_6_output_0" ||
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
  if (IsSam3MaskBridgeTensor(tensor.info.name)) {
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

std::vector<std::uint8_t> PackTextPromptBytes(const Sam3Prompt& prompt) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(prompt.text.size() + sizeof(int));
  const int length = static_cast<int>(prompt.text.size());
  AppendInt(bytes, length);
  bytes.insert(bytes.end(), prompt.text.begin(), prompt.text.end());
  return bytes;
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

void BindInputsByName(const std::string& stage_name,
                      const std::vector<const BpuTensorBuffer*>& sources,
                      std::vector<BpuTensorBuffer>& inputs) {
  for (auto& input : inputs) {
    const auto* source = FindSourceTensor(sources, input.info.name);
    if (source == nullptr) {
      if (SynthesizeSam3BridgeTensor(stage_name, input)) {
        continue;
      }
      throw std::runtime_error(stage_name + " input tensor has no upstream binding: " + input.info.name);
    }
    CopyBuffer(*source, input, stage_name);
  }
}

std::vector<BpuTensorBuffer> RunStageFromSources(const BpuModel& stage,
                                                 const std::string& stage_name,
                                                 const std::vector<const BpuTensorBuffer*>& sources) {
  auto inputs = stage.AllocateInputs();
  auto outputs = stage.AllocateOutputs();
  BindInputsByName(stage_name, sources, inputs);
  stage.Infer(inputs, outputs);
  return outputs;
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

bool HasRealSplitMaskDecoder(const Sam3Model& model) {
  return model.FindPart("mask_decoder_pre_norm") != nullptr && model.FindPart("mask_decoder_post_norm") != nullptr;
}

bool HasSmokeSplitMaskDecoder(const Sam3Model& model) {
  return model.FindPart("mask_decoder_pixel_mid") != nullptr && model.FindPart("mask_decoder_pixel2_post_norm") != nullptr;
}

void RequirePromptParts(const Sam3Model& model, const Sam3Prompt& prompt) {
  (void)RequirePart(model, "image_encoder");
  (void)RequirePart(model, "detector");
  if (!HasRealSplitMaskDecoder(model) && !HasSmokeSplitMaskDecoder(model)) {
    (void)RequirePart(model, "mask_decoder");
  }

  if (HasPromptType(prompt, Sam3PromptType::kText)) {
    (void)RequirePart(model, "text_encoder");
  }
  if (HasPromptType(prompt, Sam3PromptType::kPoint) || HasPromptType(prompt, Sam3PromptType::kBox) ||
      HasPromptType(prompt, Sam3PromptType::kMask) || HasPromptType(prompt, Sam3PromptType::kExemplar)) {
    (void)RequirePart(model, "geometry_encoder");
  }
}

}  // namespace

Sam3ImagePredictor::Sam3ImagePredictor(Sam3Model model) : model_(std::move(model)) {
  if (!model_.Loaded()) {
    model_.Load(false);
  }
}

Sam3ImageResult Sam3ImagePredictor::Predict(const Image& image, const Sam3Prompt& prompt) const {
  if (image.width <= 0 || image.height <= 0) {
    throw std::runtime_error("image dimensions are invalid");
  }

  ValidatePrompt(prompt);
  RequirePromptParts(model_, prompt);

  std::vector<BpuTensorBuffer> text_outputs;
  std::vector<BpuTensorBuffer> geometry_outputs;
  if (HasPromptType(prompt, Sam3PromptType::kText)) {
    text_outputs = RunPromptEncoder(RequirePart(model_, "text_encoder"), "text_encoder", PackTextPromptBytes(prompt));
  }
  if (HasPromptType(prompt, Sam3PromptType::kPoint) || HasPromptType(prompt, Sam3PromptType::kBox) ||
      HasPromptType(prompt, Sam3PromptType::kMask) || HasPromptType(prompt, Sam3PromptType::kExemplar)) {
    geometry_outputs = RunPromptEncoder(RequirePart(model_, "geometry_encoder"), "geometry_encoder", PackGeometryPromptBytes(prompt));
  }

  const auto& image_encoder = RequirePart(model_, "image_encoder");
  auto image_inputs = image_encoder.AllocateInputs();
  auto image_outputs = image_encoder.AllocateOutputs();
  CopyImageBytesToTensors(image, image_inputs);
  image_encoder.Infer(image_inputs, image_outputs);

  std::vector<BpuTensorBuffer> detector_tap_outputs;
  auto tap_sources = TensorRefs(image_inputs);
  AppendTensorRefs(tap_sources, image_outputs);
  AppendTensorRefs(tap_sources, text_outputs);
  AppendTensorRefs(tap_sources, geometry_outputs);
  if (const auto* detector_taps = model_.FindPart("detector_taps")) {
    detector_tap_outputs = RunStageFromSources(*detector_taps, "detector_taps", tap_sources);
  } else {
    if (const auto* detector_bridge_taps = model_.FindPart("detector_bridge_taps")) {
      detector_tap_outputs = RunStageFromSources(*detector_bridge_taps, "detector_bridge_taps", tap_sources);
    } else {
      const char* micro_tap_names[] = {
          "detector_image_bridge_taps",
          "detector_text_bridge_tap",
          "detector_geometry_bridge_taps",
      };
      for (const auto* micro_tap_name : micro_tap_names) {
        if (const auto* micro_tap = model_.FindPart(micro_tap_name)) {
          auto micro_tap_outputs = RunStageFromSources(*micro_tap, micro_tap_name, tap_sources);
          for (auto& output : micro_tap_outputs) {
            detector_tap_outputs.push_back(std::move(output));
          }
          AppendTensorRefs(tap_sources, detector_tap_outputs);
        }
      }
    }
    if (const auto* detector_encoder_hidden_tap = model_.FindPart("detector_encoder_hidden_tap")) {
      AppendTensorRefs(tap_sources, detector_tap_outputs);
      auto encoder_tap_outputs = RunStageFromSources(*detector_encoder_hidden_tap,
                                                     "detector_encoder_hidden_tap",
                                                     tap_sources);
      for (auto& output : encoder_tap_outputs) {
        detector_tap_outputs.push_back(std::move(output));
      }
    }
  }

  auto detector_sources = TensorRefs(image_outputs);
  AppendTensorRefs(detector_sources, text_outputs);
  AppendTensorRefs(detector_sources, geometry_outputs);
  AppendTensorRefs(detector_sources, detector_tap_outputs);
  auto detector_outputs = RunStageFromSources(RequirePart(model_, "detector"), "detector", detector_sources);

  auto mask_sources = TensorRefs(image_outputs);
  AppendTensorRefs(mask_sources, text_outputs);
  AppendTensorRefs(mask_sources, geometry_outputs);
  AppendTensorRefs(mask_sources, detector_tap_outputs);
  AppendTensorRefs(mask_sources, detector_outputs);
  if (HasRealSplitMaskDecoder(model_)) {
    auto pre_norm_outputs = RunStageFromSources(RequirePart(model_, "mask_decoder_pre_norm"),
                                                "mask_decoder_pre_norm",
                                                mask_sources);
    auto post_norm_inputs = RequirePart(model_, "mask_decoder_post_norm").AllocateInputs();
    if (post_norm_inputs.size() != 1) {
      throw std::runtime_error("mask_decoder_post_norm expects one input");
    }
    auto norm_output = CpuHighResolutionInstanceNorm(pre_norm_outputs.front(), post_norm_inputs.front().info);
    std::vector<const BpuTensorBuffer*> post_norm_sources{&norm_output};
    (void)RunStageFromSources(RequirePart(model_, "mask_decoder_post_norm"),
                              "mask_decoder_post_norm",
                              post_norm_sources);
  } else if (HasSmokeSplitMaskDecoder(model_)) {
    auto pixel_mid_outputs = RunStageFromSources(RequirePart(model_, "mask_decoder_pixel_mid"),
                                                 "mask_decoder_pixel_mid",
                                                 mask_sources);
    auto pixel2_sources = TensorRefs(pixel_mid_outputs);
    (void)RunStageFromSources(RequirePart(model_, "mask_decoder_pixel2_post_norm"),
                              "mask_decoder_pixel2_post_norm",
                              pixel2_sources);
  } else {
    (void)RunStageFromSources(RequirePart(model_, "mask_decoder"), "mask_decoder", mask_sources);
  }

  return Sam3ImageResult{};
}

}  // namespace sam_s600
