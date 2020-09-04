
#include "detector.hpp"

#include <NvOnnxParser.h>

#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

#include "cuda_runtime_api.h"
#include "opencv2/core/mat.hpp"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include "spdlog/spdlog.h"

using namespace nvinfer1;

cv::Mat Preprocess(cv::Mat raw) {
  cv::Mat image;
  raw.convertTo(image, CV_32FC3, 1.0 / 255.0, 0);
  return image;
}

static float IOU(const BBox &bbox1, const BBox &bbox2) {
  const float left =
      std::max(bbox1.x_ctr - bbox1.w / 2.f, bbox2.x_ctr - bbox2.w / 2.f);
  const float right =
      std::min(bbox1.x_ctr + bbox1.w / 2.f, bbox2.x_ctr + bbox2.w / 2.f);
  const float top =
      std::max(bbox1.y_ctr - bbox1.h / 2.f, bbox2.y_ctr - bbox2.h / 2.f);
  const float bottom =
      std::min(bbox1.y_ctr + bbox1.h / 2.f, bbox2.y_ctr + bbox2.h / 2.f);

  if (top > bottom || left > right) return 0.0f;

  const float inter_box_s = (right - left) * (bottom - top);
  return inter_box_s / (bbox1.w * bbox1.h + bbox2.w * bbox2.h - inter_box_s);
}

static std::vector<Detection> PostProcess(const float *prob,
                                          float conf_thresh) {
  int range = prob[0] < 1000 ? prob[0] : 1000;
  std::vector<Detection> dets(&prob[1], &prob[range - 1]);
  dets.erase(std::remove_if(dets.begin(), dets.end(),
                            [conf_thresh](const Detection &d) {
                              return (d.conf < conf_thresh);
                            }),
             dets.end());
  return dets;
}

static std::vector<Detection> NonMaxSuppression(
    const std::vector<Detection> &dets, float ovr_thresh, float neighbor_thresh,
    float score_thresh) {
  std::vector<Detection> out;

  std::multimap<float, size_t> scores_idxs;
  for (size_t i = 0; i < dets.size(); ++i)
    scores_idxs.insert(std::pair<float, size_t>(dets.at(i).conf, i));

  while (!scores_idxs.empty()) {
    auto last_scores_idxs = --scores_idxs.end();
    const auto det = dets[last_scores_idxs->second];

    int num_neigbors = 0;
    float score_sum = det.conf;

    scores_idxs.erase(last_scores_idxs);

    for (auto it = scores_idxs.begin(); it != scores_idxs.end();) {
      const auto det2 = dets[it->second];

      if (IOU(det.bbox, det2.bbox) > ovr_thresh) {
        score_sum += it->first;
        it = scores_idxs.erase(it);
        ++num_neigbors;
      } else {
        ++it;
      }
    }
    if (num_neigbors >= neighbor_thresh && score_sum >= score_thresh)
      out.push_back(det);
  }
  return out;
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

  auto network = UniquePtr<INetworkDefinition>(builder->c(explicit_batch));
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

  for (auto it = bindings_.begin(); it != bindings_.end(); ++it) cudaFree(*it);

  // camera_.Close();
  SPDLOG_DEBUG("[Detector] Destructed.");
}

bool Detector::TestInfer() {
  SPDLOG_DEBUG("[Detector] TestInfer.");
  cv::Mat image = cv::imread("./image/test.jpg");
  cv::resize(image, image, cv::Size(608, 608));

  std::vector<float> output(bingings_size_.at(idx_out_) / sizeof(float));

  cudaMemcpy(bindings_.at(idx_in_), image.data, bingings_size_.at(idx_in_),
             cudaMemcpyHostToDevice);
  context_->executeV2(bindings_.data());
  cudaMemcpy(output.data(), bindings_.at(idx_out_), bingings_size_.at(idx_out_),
             cudaMemcpyDeviceToHost);

  auto dets = PostProcess(output.data(), conf_thres_);
  auto final = NonMaxSuppression(dets, 0.5, neighbor_thresh_, conf_thres_);

  for (auto it = final.begin(); it != final.end(); ++it) {
    const cv::Point org(it->bbox.x_ctr - it->bbox.w / 2,
                        it->bbox.y_ctr - it->bbox.h / 2);
    const cv::Size s(it->bbox.w, it->bbox.h);
    const cv::Rect roi(org, s);
    cv::rectangle(image, roi, cv::Scalar(0, 255, 0));
    cv::putText(image, std::to_string(it->class_id), org,
                cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 2.0, cv::Scalar(0, 255, 0));
  }

  cv::imwrite("./image/result/test_tensorrt.jpg", image);

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

  auto dets = PostProcess(output.data(), conf_thres_);
  auto final = NonMaxSuppression(dets, 0.5, neighbor_thresh_, conf_thres_);

  SPDLOG_DEBUG("[Detector] Infered.");
  return final;
}
