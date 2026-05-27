#include "sam_s600/sam3/sam3_image_predictor.hpp"

#include <algorithm>
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

void CopyImageBytesToTensors(const Image& image, std::vector<BpuTensorBuffer>& inputs) {
  if (image.data.empty()) {
    throw std::runtime_error("image data is empty");
  }
  if (inputs.empty()) {
    throw std::runtime_error("image encoder has no inputs");
  }

  const std::size_t required_bytes = TotalTensorBytes(inputs);
  if (image.data.size() != required_bytes) {
    throw std::runtime_error("image data size does not match SAM3 image encoder inputs");
  }

  std::size_t offset = 0;
  for (auto& input : inputs) {
    auto* dst = input.buffer.CpuData();
    if (dst == nullptr) {
      throw std::runtime_error("image encoder input buffer has no CPU address");
    }
    std::copy_n(image.data.data() + offset, input.buffer.Size(), dst);
    input.buffer.CleanCache();
    offset += input.buffer.Size();
  }
}

void RequirePromptParts(const Sam3Model& model, const Sam3Prompt& prompt) {
  (void)RequirePart(model, "image_encoder");
  (void)RequirePart(model, "detector");
  (void)RequirePart(model, "mask_decoder");

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

  const auto& image_encoder = RequirePart(model_, "image_encoder");
  auto inputs = image_encoder.AllocateInputs();
  auto outputs = image_encoder.AllocateOutputs();
  CopyImageBytesToTensors(image, inputs);
  image_encoder.Infer(inputs, outputs);

  return Sam3ImageResult{};
}

}  // namespace sam_s600
