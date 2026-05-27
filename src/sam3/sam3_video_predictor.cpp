#include "sam_s600/sam3/sam3_video_predictor.hpp"

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

std::size_t TotalTensorBytes(const std::vector<BpuTensorBuffer>& buffers) {
  std::size_t total = 0;
  for (const auto& buffer : buffers) {
    total += buffer.buffer.Size();
  }
  return total;
}

void CopyFrameBytesToTensors(const VideoFrame& frame, std::vector<BpuTensorBuffer>& inputs) {
  if (frame.image.data.empty()) {
    throw std::runtime_error("video frame image data is empty");
  }
  if (inputs.empty()) {
    throw std::runtime_error("image encoder has no inputs");
  }

  const std::size_t required_bytes = TotalTensorBytes(inputs);
  if (frame.image.data.size() != required_bytes) {
    throw std::runtime_error("video frame data size does not match SAM3 image encoder inputs");
  }

  std::size_t offset = 0;
  for (auto& input : inputs) {
    auto* dst = input.buffer.CpuData();
    if (dst == nullptr) {
      throw std::runtime_error("image encoder input buffer has no CPU address");
    }
    std::copy_n(frame.image.data.data() + offset, input.buffer.Size(), dst);
    input.buffer.CleanCache();
    offset += input.buffer.Size();
  }
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

void RequireVideoParts(const Sam3Model& model, const Sam3Prompt& prompt) {
  (void)RequirePart(model, "image_encoder");
  (void)RequirePart(model, "detector");
  (void)RequirePart(model, "mask_decoder");
  (void)RequirePart(model, "memory_encoder");
  (void)RequirePart(model, "tracker");

  if (HasPromptType(prompt, Sam3PromptType::kText)) {
    (void)RequirePart(model, "text_encoder");
  }
  if (HasPromptType(prompt, Sam3PromptType::kPoint) || HasPromptType(prompt, Sam3PromptType::kBox) ||
      HasPromptType(prompt, Sam3PromptType::kMask) || HasPromptType(prompt, Sam3PromptType::kExemplar)) {
    (void)RequirePart(model, "geometry_encoder");
  }
}

}  // namespace

Sam3VideoPredictor::Sam3VideoPredictor(Sam3Model model) : model_(std::move(model)) {
  if (!model_.Loaded()) {
    model_.Load(false);
  }
}

Sam3VideoResult Sam3VideoPredictor::Predict(const std::vector<VideoFrame>& frames, const Sam3Prompt& prompt) const {
  if (frames.empty()) {
    throw std::runtime_error("video frame list is empty");
  }

  ValidatePrompt(prompt);
  RequireVideoParts(model_, prompt);

  const auto& image_encoder = RequirePart(model_, "image_encoder");
  Sam3VideoResult result;
  result.frames.reserve(frames.size());

  for (const auto& frame : frames) {
    if (frame.image.width <= 0 || frame.image.height <= 0) {
      throw std::runtime_error("video frame image dimensions are invalid");
    }

    auto inputs = image_encoder.AllocateInputs();
    auto outputs = image_encoder.AllocateOutputs();
    CopyFrameBytesToTensors(frame, inputs);
    image_encoder.Infer(inputs, outputs);

    Sam3VideoFrameResult frame_result;
    frame_result.pts_us = frame.pts_us;
    result.frames.push_back(std::move(frame_result));
  }

  return result;
}

}  // namespace sam_s600
