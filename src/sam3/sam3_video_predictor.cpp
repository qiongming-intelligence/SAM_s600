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

const BpuTensorBuffer* FindTensor(const std::vector<const BpuTensorBuffer*>& sources, const std::string& name) {
  for (const auto* source : sources) {
    if (source != nullptr && source->info.name == name) {
      return source;
    }
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
    const auto* source = FindTensor(sources, input.info.name);
    if (source == nullptr) {
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
  const auto& detector = RequirePart(model_, "detector");
  const auto& mask_decoder = RequirePart(model_, "mask_decoder");
  const auto& tracker = RequirePart(model_, "tracker");
  const auto& memory_encoder = RequirePart(model_, "memory_encoder");
  std::vector<BpuTensorBuffer> memory_outputs;
  Sam3VideoResult result;
  result.frames.reserve(frames.size());

  for (const auto& frame : frames) {
    if (frame.image.width <= 0 || frame.image.height <= 0) {
      throw std::runtime_error("video frame image dimensions are invalid");
    }

    auto image_inputs = image_encoder.AllocateInputs();
    auto image_outputs = image_encoder.AllocateOutputs();
    CopyFrameBytesToTensors(frame, image_inputs);
    image_encoder.Infer(image_inputs, image_outputs);

    const auto detector_sources = TensorRefs(image_outputs);
    auto detector_outputs = RunStageFromSources(detector, "detector", detector_sources);

    auto mask_sources = TensorRefs(image_outputs);
    AppendTensorRefs(mask_sources, detector_outputs);
    auto mask_outputs = RunStageFromSources(mask_decoder, "mask_decoder", mask_sources);

    auto tracker_sources = TensorRefs(image_outputs);
    AppendTensorRefs(tracker_sources, detector_outputs);
    AppendTensorRefs(tracker_sources, mask_outputs);
    AppendTensorRefs(tracker_sources, memory_outputs);
    auto tracker_outputs = RunStageFromSources(tracker, "tracker", tracker_sources);

    auto memory_sources = TensorRefs(image_outputs);
    AppendTensorRefs(memory_sources, detector_outputs);
    AppendTensorRefs(memory_sources, mask_outputs);
    AppendTensorRefs(memory_sources, tracker_outputs);
    memory_outputs = RunStageFromSources(memory_encoder, "memory_encoder", memory_sources);

    Sam3VideoFrameResult frame_result;
    frame_result.pts_us = frame.pts_us;
    result.frames.push_back(std::move(frame_result));
  }

  return result;
}

}  // namespace sam_s600
