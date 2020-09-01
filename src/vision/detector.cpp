#include "obj_detector.hpp"

#include <NvOnnxParser.h>

#include <fstream>
#include <stdexcept>

#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include "spdlog/spdlog.h"

#if 1
using namespace nvinfer1;

template <typename T>
void TRTDeleter::operator()(T *obj) const {
  if (obj) {
    SPDLOG_DEBUG("[TRTDeleter] destroy.");
    obj->destroy();
  }
}

void TRTLogger::log(Severity severity, const char *msg) {
  if (severity == Severity::kINTERNAL_ERROR) {
    spdlog::error(msg);
  } else if (severity == Severity::kERROR) {
    spdlog::error(msg);
  } else if (severity == Severity::kWARNING) {
    spdlog::warn(msg);
  } else if (severity == Severity::kINFO) {
    spdlog::info(msg);
  } else if (severity == Severity::kVERBOSE) {
    spdlog::debug(msg);
  }
}

int TRTLogger::GetVerbosity() { return (int)Severity::kVERBOSE; }

Object::Object(x_center, y_center, width, height)
    : x_center_{x_center},
      y_center_{y_center},
      width_{width},
      height_{height} {}

cv::Point Object::Center() { return cv::Point(x_center_, y_center_); }

cv::Rect Object::Rect() {
  return cv::Rect(x_center_, y_center_, width_, height_);
}

bool Detector::Preprocesse() {
  // image /= 255.0
  return true;
}

bool Detector::CreateEngine() {
  SPDLOG_DEBUG("[Detector] CreateEngine.");

  auto builder = UniquePtr<IBuilder>(createInferBuilder(logger_));
  if (!builder) {
    SPDLOG_ERROR("[Detector] createInferBuilder Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[Detector] createInferBuilder OK.");

  builder->setMaxBatchSize(1);

  const auto explicit_batch =
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network =
      UniquePtr<INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) {
    SPDLOG_ERROR("[Detector] createNetworkV2 Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[Detector] createNetworkV2 OK.");

  auto config = UniquePtr<IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    SPDLOG_ERROR("[Detector] createBuilderConfig Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[Detector] createBuilderConfig OK.");

  config->setMaxWorkspaceSize(1 << 30);

  auto parser = UniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger_));
  if (!parser) {
    SPDLOG_ERROR("[Detector] createParser Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[Detector] createParser OK.");

  auto parsed = parser->parseFromFile(onnx_file_path_.c_str(),
                                      static_cast<int>(logger_.GetVerbosity()));
  if (!parsed) {
    SPDLOG_ERROR("[Detector] parseFromFile Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[Detector] parseFromFile OK.");

  auto profile = builder->createOptimizationProfile();
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kMIN, Dims4{1, 3, 608, 608});
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kOPT, Dims4{1, 3, 608, 608});
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kMAX, Dims4{1, 3, 608, 608});
  config->addOptimizationProfile(profile);

  if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);
  if (builder->platformHasFastInt8()) config->setFlag(BuilderFlag::kINT8);

  if (builder->getNbDLACores() == 0)
    SPDLOG_WARN("[Detector] The platform doesn't have any DLA cores.");
  else {
    SPDLOG_INFO("[Detector] Using DLA core 0.");
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(0);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
  }

  SPDLOG_INFO("[Detector] CreateEngine, please wait for a while...");

  engine_ = std::shared_ptr<ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config), TRTDeleter());

  if (!engine_) {
    SPDLOG_ERROR("[Detector] CreateEngine Fail.");
    return false;
  }
  SPDLOG_INFO("[Detector] CreateEngine OK.");
  return true;
}

bool Detector::LoadEngine() {
  SPDLOG_DEBUG("[Detector] LoadEngine.");

  std::vector<char> engine_bin;
  std::ifstream engine_file(engine_path_, std::ios::binary);

  if (engine_file.good()) {
    engine_file.seekg(0, engine_file.end);
    engine_bin.resize(engine_file.tellg());
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(engine_bin.data(), engine_bin.size());
    engine_file.close();
  } else {
    SPDLOG_ERROR("[Detector] LoadEngine Fail. Could not open file.");
    return false;
  }

  auto runtime = UniquePtr<IRuntime>(createInferRuntime(logger_));

  engine_ = std::shared_ptr<ICudaEngine>(
      runtime->deserializeCudaEngine(engine_bin.data(), engine_bin.size()),
      TRTDeleter());

  if (!engine_) {
    SPDLOG_ERROR("[Detector] LoadEngine Fail.");
    return false;
  }
  SPDLOG_DEBUG("[Detector] LoadEngine OK.");
  return true;
}

bool Detector::SaveEngine() {
  SPDLOG_ERROR("[Detector] SaveEngine.");

  if (engine_) {
    auto engine_serialized = UniquePtr<IHostMemory>(engine_->serialize());
    std::ofstream engine_file(engine_path_, std::ios::binary);
    if (!engine_file) {
      SPDLOG_ERROR("[Detector] SaveEngine Fail. Could not open file.");
      return false;
    }
    engine_file.write(reinterpret_cast<const char *>(engine_serialized->data()),
                      engine_serialized->size());

    SPDLOG_DEBUG("[Detector] SaveEngine OK.");
    return true;
  }
  SPDLOG_ERROR("[Detector] SaveEngine Fail. No engine_.");
  return false;
}

bool Detector::CreateContex() {
  SPDLOG_DEBUG("[Detector] CreateContex.");
  context_ = std::shared_ptr<IExecutionContext>(engine_->createExecutionContext(),
                                               TRTDeleter());
  if (!context_) {
    SPDLOG_ERROR("[Detector] CreateContex Fail.");
    return false;
  }
  SPDLOG_DEBUG("[Detector] CreateContex OK.");
  return true;
}

bool Detector::InitMemory() {
  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    Dims dim = engine_->getBindingDimensions(i);

    size_t volume = 1;
    for (int j = 0; j < dim.nbDims; ++j) volume *= dim.d[j];
    nvinfer1::DataType type = engine_->getBindingDataType(i);
    switch (type) {
      case DataType::kFLOAT:
        volume *= sizeof(float);
        break;

      default:
        SPDLOG_ERROR("[Detector] Do not support input type: {}", type);
        throw std::runtime_error("[Detector] Unsupported input type");
        break;
    }

    void *device_memory;
    cudaMalloc(&device_memory, volume);
    bindings_.push_back(device_memory);

    SPDLOG_DEBUG("[Detector] Binding {} : {}", i,
                 engine_->getBindingName(i));
  }
  idx_in_ = engine_->getBindingIndex("images");
  idx_out_ = engine_->getBindingIndex("output");
  return true;
}

Detector::Detector() {
  SPDLOG_DEBUG("[Detector] Constructing.");
  onnx_file_path_ = "./best.onnx";
  engine_path_ = onnx_file_path_ + ".engine_";

  // camera_.Open(index);
  if (!LoadEngine()) {
    CreateEngine();
    SaveEngine();
  }
  CreateContex();
  InitMemory();
  SPDLOG_DEBUG("[Detector] Constructed.");
}

Detector::~Detector() {
  SPDLOG_DEBUG("[Detector] Destructing.");

  for (std::vector<void *>::iterator it = bindings_.begin();
       it != bindings_.end(); ++it)
    cudaFree(*it);

  // camera_.Close();
  SPDLOG_DEBUG("[Detector] Destructed.");
}

bool Detector::TestInfer() {
  SPDLOG_DEBUG("[Detector] TestInfer.");
  cv::Mat image = cv::imread("./image/test.jpg");
  cv::resize(image, image, cv::Size(608, 608));

  std::vector<float> output;
  output.resize(1000);

  cudaMemcpy(bindings_.at(idx_in_), image.data, 608 * 608 * 3 * sizeof(float),
             cudaMemcpyHostToDevice);
  context_->executeV2(bindings_.data());
  cudaMemcpy(output.data(), bindings_.at(idx_out_), 10000,
             cudaMemcpyDeviceToHost);

  for (std::vector<float>::iterator it = output.begin(); it != output.end();
       ++it) {
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
    cv::rectangle(image, rect, cv::Scalar(0, 255, 0));
    cv::putText(image, std::to_string(), );
  }

  cv::imwrite("./image/result/test_tensorrt.jpg", image);

  SPDLOG_DEBUG("[Detector] TestInfer done.");
  return true;
}

bool Detector::Infer() {
  SPDLOG_DEBUG("[Detector] Infer.");

  // Get frame from camera_.
  // preprocessing image.
  // Do infer.
  // processing result.
  // result output.

  SPDLOG_DEBUG("[Detector] Infered.");
  return true;
}

#endif
