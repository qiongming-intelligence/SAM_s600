#include "sam_s600/multiplex/multiplex_video_predictor.hpp"

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

void CopyFrameBytesToTensors(const VideoFrame& frame, std::vector<BpuTensorBuffer>& inputs) {
  if (frame.image.data.empty()) {
    throw std::runtime_error("multiplex video frame image data is empty");
  }
  if (inputs.empty()) {
    throw std::runtime_error("multiplex detector has no inputs");
  }

  const std::size_t required_bytes = TotalTensorBytes(inputs);
  if (frame.image.data.size() != required_bytes) {
    throw std::runtime_error("video frame data size does not match SAM3 multiplex detector inputs");
  }

  std::size_t offset = 0;
  for (auto& input : inputs) {
    auto* dst = input.buffer.CpuData();
    if (dst == nullptr) {
      throw std::runtime_error("multiplex detector input buffer has no CPU address");
    }
    std::copy_n(frame.image.data.data() + offset, input.buffer.Size(), dst);
    input.buffer.CleanCache();
    offset += input.buffer.Size();
  }
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

void RequireMultiplexVideoParts(const Sam3Model& model) {
  (void)RequirePart(model, "multiplex_detector");
  (void)RequirePart(model, "multiplex_tracker");
  (void)RequirePart(model, "memory_encoder");
}

}  // namespace

Sam3MultiplexVideoPredictor::Sam3MultiplexVideoPredictor(Sam3Model model) : model_(std::move(model)) {
  if (!model_.Loaded()) {
    model_.Load(false);
  }
}

Sam3VideoResult Sam3MultiplexVideoPredictor::Predict(const std::vector<VideoFrame>& frames,
                                                     const Sam3Prompt& prompt) const {
  if (frames.empty()) {
    throw std::runtime_error("multiplex video frame list is empty");
  }

  ValidatePrompt(prompt);
  RequireMultiplexVideoParts(model_);

  const auto& detector = RequirePart(model_, "multiplex_detector");
  const auto& tracker = RequirePart(model_, "multiplex_tracker");
  const auto& memory_encoder = RequirePart(model_, "memory_encoder");

  Sam3VideoResult result;
  result.frames.reserve(frames.size());

  for (const auto& frame : frames) {
    if (frame.image.width <= 0 || frame.image.height <= 0) {
      throw std::runtime_error("multiplex video frame image dimensions are invalid");
    }

    auto detector_inputs = detector.AllocateInputs();
    auto detector_outputs = detector.AllocateOutputs();
    CopyFrameBytesToTensors(frame, detector_inputs);
    detector.Infer(detector_inputs, detector_outputs);

    const auto tracker_sources = TensorRefs(detector_outputs);
    auto tracker_outputs = RunStageFromSources(tracker, "multiplex_tracker", tracker_sources);

    auto memory_sources = TensorRefs(detector_outputs);
    AppendTensorRefs(memory_sources, tracker_outputs);
    (void)RunStageFromSources(memory_encoder, "memory_encoder", memory_sources);

    Sam3VideoFrameResult frame_result;
    frame_result.pts_us = frame.pts_us;
    result.frames.push_back(std::move(frame_result));
  }

  return result;
}

}  // namespace sam_s600
