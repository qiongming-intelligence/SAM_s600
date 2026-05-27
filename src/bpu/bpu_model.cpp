#include "sam_s600/bpu/bpu_model.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(SAM_S600_HAS_HOBOT_DNN)
#include <hobot/dnn/hb_dnn.h>
#include <hobot/dnn/hb_dnn_status.h>
#endif

namespace sam_s600 {
namespace {

std::string Basename(const std::string& path) {
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

#if defined(SAM_S600_HAS_HOBOT_DNN)

void CheckDnn(int32_t code, const std::string& action) {
  if (code != 0) {
    throw std::runtime_error(action + " failed: " + hbDNNGetErrorDesc(code));
  }
}

TensorDataType ConvertDnnType(int32_t type) {
  switch (type) {
    case HB_DNN_TENSOR_TYPE_S4:
      return TensorDataType::kInt4;
    case HB_DNN_TENSOR_TYPE_U4:
      return TensorDataType::kUint4;
    case HB_DNN_TENSOR_TYPE_S8:
      return TensorDataType::kInt8;
    case HB_DNN_TENSOR_TYPE_U8:
      return TensorDataType::kUint8;
    case HB_DNN_TENSOR_TYPE_F16:
      return TensorDataType::kFloat16;
    case HB_DNN_TENSOR_TYPE_S16:
      return TensorDataType::kInt16;
    case HB_DNN_TENSOR_TYPE_U16:
      return TensorDataType::kUint16;
    case HB_DNN_TENSOR_TYPE_F32:
      return TensorDataType::kFloat32;
    case HB_DNN_TENSOR_TYPE_S32:
      return TensorDataType::kInt32;
    case HB_DNN_TENSOR_TYPE_U32:
      return TensorDataType::kUint32;
    case HB_DNN_TENSOR_TYPE_F64:
      return TensorDataType::kFloat64;
    case HB_DNN_TENSOR_TYPE_S64:
      return TensorDataType::kInt64;
    case HB_DNN_TENSOR_TYPE_U64:
      return TensorDataType::kUint64;
    case HB_DNN_TENSOR_TYPE_BOOL8:
      return TensorDataType::kBool8;
    default:
      return TensorDataType::kUnknown;
  }
}

TensorInfo TensorInfoFromDnn(const char* name, const hbDNNTensorProperties& properties) {
  TensorInfo info;
  info.name = name == nullptr ? std::string{} : std::string{name};
  info.dtype = ConvertDnnType(properties.tensorType);
  info.byte_size = properties.alignedByteSize < 0 ? 0 : static_cast<std::uint64_t>(properties.alignedByteSize);
  for (int32_t i = 0; i < properties.validShape.numDimensions; ++i) {
    info.shape.dims.push_back(properties.validShape.dimensionSize[i]);
    info.stride.push_back(properties.stride[i]);
  }
  return info;
}

#endif

}  // namespace

struct BpuModel::Impl {
#if defined(SAM_S600_HAS_HOBOT_DNN)
  hbDNNPackedHandle_t packed_handle{nullptr};
  hbDNNHandle_t model_handle{nullptr};
#endif
};

BpuModel::BpuModel() : impl_(std::make_unique<Impl>()) {}

BpuModel::BpuModel(std::string model_path) : BpuModel() { Load(std::move(model_path)); }

BpuModel::~BpuModel() { Reset(); }

BpuModel::BpuModel(BpuModel&&) noexcept = default;

BpuModel& BpuModel::operator=(BpuModel&& other) noexcept {
  if (this != &other) {
    Reset();
    impl_ = std::move(other.impl_);
    path_ = std::move(other.path_);
    name_ = std::move(other.name_);
    inputs_ = std::move(other.inputs_);
    outputs_ = std::move(other.outputs_);
    compile_bpu_core_num_ = other.compile_bpu_core_num_;
    loaded_ = other.loaded_;
    other.compile_bpu_core_num_ = 0;
    other.loaded_ = false;
  }
  return *this;
}

void BpuModel::Load(std::string model_path) {
  Reset();
  path_ = std::move(model_path);
  name_ = Basename(path_);

  if (path_.empty()) {
    throw std::runtime_error("BPU model path is empty");
  }
  if (!std::filesystem::exists(path_)) {
    throw std::runtime_error("BPU model file does not exist: " + path_);
  }

#if defined(SAM_S600_HAS_HOBOT_DNN)
  const char* files[] = {path_.c_str()};
  CheckDnn(hbDNNInitializeFromFiles(&impl_->packed_handle, files, 1), "hbDNNInitializeFromFiles");

  char const** model_names = nullptr;
  int32_t model_count = 0;
  CheckDnn(hbDNNGetModelNameList(&model_names, &model_count, impl_->packed_handle), "hbDNNGetModelNameList");
  if (model_count <= 0 || model_names == nullptr || model_names[0] == nullptr) {
    throw std::runtime_error("HBM file contains no DNN models: " + path_);
  }

  name_ = model_names[0];
  CheckDnn(hbDNNGetModelHandle(&impl_->model_handle, impl_->packed_handle, name_.c_str()), "hbDNNGetModelHandle");

  int32_t input_count = 0;
  CheckDnn(hbDNNGetInputCount(&input_count, impl_->model_handle), "hbDNNGetInputCount");
  for (int32_t i = 0; i < input_count; ++i) {
    char const* input_name = nullptr;
    hbDNNTensorProperties properties{};
    CheckDnn(hbDNNGetInputName(&input_name, impl_->model_handle, i), "hbDNNGetInputName");
    CheckDnn(hbDNNGetInputTensorProperties(&properties, impl_->model_handle, i), "hbDNNGetInputTensorProperties");
    inputs_.push_back(TensorInfoFromDnn(input_name, properties));
  }

  int32_t output_count = 0;
  CheckDnn(hbDNNGetOutputCount(&output_count, impl_->model_handle), "hbDNNGetOutputCount");
  for (int32_t i = 0; i < output_count; ++i) {
    char const* output_name = nullptr;
    hbDNNTensorProperties properties{};
    CheckDnn(hbDNNGetOutputName(&output_name, impl_->model_handle, i), "hbDNNGetOutputName");
    CheckDnn(hbDNNGetOutputTensorProperties(&properties, impl_->model_handle, i), "hbDNNGetOutputTensorProperties");
    outputs_.push_back(TensorInfoFromDnn(output_name, properties));
  }

  CheckDnn(hbDNNGetCompileBpuCoreNum(&compile_bpu_core_num_, impl_->model_handle), "hbDNNGetCompileBpuCoreNum");
#else
  compile_bpu_core_num_ = 0;
#endif

  loaded_ = true;
}

void BpuModel::Reset() {
#if defined(SAM_S600_HAS_HOBOT_DNN)
  if (impl_ && impl_->packed_handle != nullptr) {
    hbDNNRelease(impl_->packed_handle);
    impl_->packed_handle = nullptr;
    impl_->model_handle = nullptr;
  }
#endif
  inputs_.clear();
  outputs_.clear();
  compile_bpu_core_num_ = 0;
  path_.clear();
  name_.clear();
  loaded_ = false;
}

bool BpuModel::Loaded() const { return loaded_; }

const std::string& BpuModel::Path() const { return path_; }

const std::string& BpuModel::Name() const { return name_; }

const std::vector<TensorInfo>& BpuModel::Inputs() const { return inputs_; }

const std::vector<TensorInfo>& BpuModel::Outputs() const { return outputs_; }

int BpuModel::CompileBpuCoreNum() const { return compile_bpu_core_num_; }

}  // namespace sam_s600
