
#include "detector.hpp"

#include <NvOnnxParser.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include "spdlog/spdlog.h"

using namespace nvinfer1;

cv::Mat Preprocess(cv::Mat raw) {
  cv::Mat image;
  raw.convertTo(image, CV_32FC3, 1.0 / 255.0, 0);
  std::vector<cv::Mat1f> ch(3);
  cv::split(image,  ch);
  cv::Mat final;
  cv::merge(final, ch);
  return final;
}

static float IOU(const Detection &det1, const Detection &det2) {
  const float left =
      std::max(det1.x_ctr - det1.w / 2.f, det2.x_ctr - det2.w / 2.f);
  const float right =
      std::min(det1.x_ctr + det1.w / 2.f, det2.x_ctr + det2.w / 2.f);
  const float top =
      std::max(det1.y_ctr - det1.h / 2.f, det2.y_ctr - det2.h / 2.f);
  const float bottom =
      std::min(det1.y_ctr + det1.h / 2.f, det2.y_ctr + det2.h / 2.f);

  if (top > bottom || left > right) return 0.0f;

  const float inter_box_s = (right - left) * (bottom - top);
  return inter_box_s / (det1.w * det1.h + det2.w * det2.h - inter_box_s);
}

void NonMaxSuppression(std::vector<Detection> &dets, float nms_thresh) {
  if (dets.empty()) return;

  std::vector<Detection> keep;

  std::sort(dets.begin(), dets.end(),
            [](const Detection &det1, const Detection &det2) {
              return det1.conf < det2.conf;
            });

  while (!dets.empty()) {
    auto highest = dets.back();
    keep.push_back(highest);
    dets.pop_back();

    for (auto it = dets.begin(); it != dets.end(); ++it) {
      if (IOU(highest, *it) > nms_thresh) {
        dets.erase(it);
      }
    }
  }
  dets = keep;
}

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

std::vector<Detection> Detector::PostProcess(std::vector<float> prob) {
  std::vector<Detection> dets;

  for (auto it = prob.begin(); it != prob.end(); it += dim_out_.d[4]) {
    if (*(it + 4) > conf_thresh_) {
      auto max_conf = std::max_element(it + 4, it + dim_out_.d[4]);
      auto class_id = std::distance(it + 4, max_conf);

      Detection det {
        *it,
        *(it + 1),
        *(it + 2),
        *(it + 3),
        *max_conf **(it + 4),
        static_cast<float>(class_id),
      };
      dets.push_back(det);
    }
  }
  // [4] obj conf
  // [5] [6] [7] [8] class_conf
  // [5] [6] [7] [8] *= [4] conf = obj_conf * cls_conf;
  // [0] [1] [2] [3] = x_ctr y_ctr w h
  return dets;
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
  // if (builder->platformHasFastInt8()) config->setFlag(BuilderFlag::kINT8);

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

  engine_ =
      UniquePtr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

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

  engine_ = UniquePtr<ICudaEngine>(
      runtime->deserializeCudaEngine(engine_bin.data(), engine_bin.size()));

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
  context_ = UniquePtr<IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    SPDLOG_ERROR("[Detector] CreateContex Fail.");
    return false;
  }
  SPDLOG_DEBUG("[Detector] CreateContex OK.");
  return true;
}

bool Detector::InitMemory() {
  idx_in_ = engine_->getBindingIndex("images");
  idx_out_ = engine_->getBindingIndex("output");
  dim_in_ = engine_->getBindingDimensions(idx_in_);
  dim_out_ = engine_->getBindingDimensions(idx_out_);
  nc = dim_out_.d[4] - 5;

  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    Dims dim = engine_->getBindingDimensions(i);

    size_t volume = 1;
    for (int j = 0; j < dim.nbDims; ++j) volume *= dim.d[j];
    DataType type = engine_->getBindingDataType(i);
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
    bingings_size_.push_back(volume);

    SPDLOG_DEBUG("[Detector] Binding {} : {}", i, engine_->getBindingName(i));
  }
  return true;
}

Detector::Detector() {
  SPDLOG_DEBUG("[Detector] Constructing.");
  onnx_file_path_ = "./mid/best.onnx";
  engine_path_ = onnx_file_path_ + ".engine";
  conf_thresh_ = 0.5;
  nms_thresh_ = 0.5;
  if (!LoadEngine()) {
    CreateEngine();
    SaveEngine();
  }
  CreateContex();
  InitMemory();
  camera_.Setup(dim_in_.d[2], dim_in_.d[3]);
  camera_.Open(0);
  SPDLOG_DEBUG("[Detector] Constructed.");
}

Detector::~Detector() {
  SPDLOG_DEBUG("[Detector] Destructing.");

  for (auto it = bindings_.begin(); it != bindings_.end(); ++it) cudaFree(*it);

  camera_.Close();
  SPDLOG_DEBUG("[Detector] Destructed.");
}

bool Detector::TestInfer() {
  SPDLOG_DEBUG("[Detector] TestInfer.");
  cv::Mat image = cv::imread("./image/test.jpg");
  cv::resize(image, image, cv::Size(608, 608));
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::imwrite("./image/test_tensorrt_in.jpg", image);

  // image = Preprocess(image);

  std::vector<float> output(bingings_size_.at(idx_out_) / sizeof(float));

  cudaMemcpy(bindings_.at(idx_in_), image.data, bingings_size_.at(idx_in_),
             cudaMemcpyHostToDevice);
  context_->executeV2(bindings_.data());
  cudaMemcpy(output.data(), bindings_.at(idx_out_), bingings_size_.at(idx_out_),
             cudaMemcpyDeviceToHost);

  auto dets = PostProcess(output);
  NonMaxSuppression(dets, nms_thresh_);

  for (auto it = dets.begin(); it != dets.end(); ++it) {
    const cv::Point org(it->x_ctr - it->w / 2, it->y_ctr - it->h / 2);
    const cv::Size s(it->w, it->h);
    const cv::Rect roi(org, s);
    cv::rectangle(image, roi, cv::Scalar(0, 255, 0));
    cv::putText(image, std::to_string(it->class_id), org,
                cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 2.0, cv::Scalar(0, 255, 0));
  }

  cv::imwrite("./image/test_tensorrt.jpg", image);

  SPDLOG_DEBUG("[Detector] TestInfer done.");
  return true;
}

std::vector<Detection> Detector::Infer() {
  SPDLOG_DEBUG("[Detector] Infer.");

  std::vector<float> output(bingings_size_.at(idx_out_) / sizeof(float));
  auto raw = camera_.GetFrame();
  auto image = Preprocess(raw);

  cudaMemcpy(bindings_.at(idx_in_), image.data, bingings_size_.at(idx_in_),
             cudaMemcpyHostToDevice);
  context_->executeV2(bindings_.data());
  cudaMemcpy(output.data(), bindings_.at(idx_out_), bingings_size_.at(idx_out_),
             cudaMemcpyDeviceToHost);

  auto dets = PostProcess(output);
  NonMaxSuppression(dets, nms_thresh_);

  SPDLOG_DEBUG("[Detector] Infered.");
  return dets;
}
