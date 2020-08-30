#include "obj_detector.hpp"

#include <NvOnnxParser.h>

#include <fstream>
#include <stdexcept>
#include <vector>

#include "cuda_runtime_api.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include "spdlog/spdlog.h"

#if 1
using namespace nvinfer1;

template <typename T>
void InferDeleter::operator()(T *obj) const {
  if (obj) {
    SPDLOG_DEBUG("[InferDeleter] destroy.");
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

bool ObjectDetector::ProcessInput() {
  // const int in_height = dim_in.d[2];
  // const int in_width = dim_in.d[3];

  // // Read a random digit file
  // srand(unsigned(time(nullptr)));
  // std::vector<uint8_t> fileData(in_height * in_width);
  // mNumber = rand() % 10;
  // readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs),
  //             fileData.data(), in_height, in_width);

  // float *hostDataBuffer =
  //     static_cast<float
  //     *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
  // for (int i = 0; i < in_height * in_width; i++) {
  //   hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
  // }

  return true;
}

bool ObjectDetector::Preprocesse() { return true; }

bool ObjectDetector::CreateEngine() {
  SPDLOG_DEBUG("[ObjectDetector] CreateEngine.");

  auto builder = UniquePtr<IBuilder>(createInferBuilder(logger));
  if (!builder) {
    SPDLOG_ERROR("[ObjectDetector] createInferBuilder Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[ObjectDetector] createInferBuilder OK.");

  builder->setMaxBatchSize(1);

  const auto explicit_batch =
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network =
      UniquePtr<INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) {
    SPDLOG_ERROR("[ObjectDetector] createNetworkV2 Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[ObjectDetector] createNetworkV2 OK.");

  auto config = UniquePtr<IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    SPDLOG_ERROR("[ObjectDetector] createBuilderConfig Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[ObjectDetector] createBuilderConfig OK.");

  config->setMaxWorkspaceSize(1 << 30);

  auto parser = UniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));
  if (!parser) {
    SPDLOG_ERROR("[ObjectDetector] createParser Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[ObjectDetector] createParser OK.");

  auto parsed = parser->parseFromFile(onnx_file_path.c_str(),
                                      static_cast<int>(logger.GetVerbosity()));
  if (!parsed) {
    SPDLOG_ERROR("[ObjectDetector] parseFromFile Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[ObjectDetector] parseFromFile OK.");

  auto profile = builder->createOptimizationProfile();
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kMIN, Dims4{1, 3, 608, 608});
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kOPT, Dims4{1, 3, 608, 608});
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kMAX, Dims4{1, 3, 608, 608});
  config->addOptimizationProfile(profile);

  dim_in = network->getInput(0)->getDimensions();
  dim_out = network->getOutput(0)->getDimensions();

  SPDLOG_INFO("[ObjectDetector] dim_in: {}, dim_out: {}.", dim_in.nbDims,
              dim_out.nbDims);

  if (use_fp16) config->setFlag(BuilderFlag::kFP16);

  if (use_int8) {
    config->setFlag(BuilderFlag::kINT8);
    // setAllTensorScales(network.get(), 127.0f, 127.0f);
  }

  if (builder->getNbDLACores() == 0)
    SPDLOG_WARN("The platform that doesn't have any DLA cores");
  else {
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    if (!config->getFlag(BuilderFlag::kINT8) && builder->platformHasFastFp16())
      config->setFlag(BuilderFlag::kFP16);

    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(0);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
  }

  SPDLOG_INFO("[ObjectDetector] CreateEngine, please wait for a while...");

  engine = std::shared_ptr<ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config), InferDeleter());

  if (!engine) {
    SPDLOG_ERROR("[ObjectDetector] CreateEngine Fail.");
    return false;
  }
  SPDLOG_INFO("[ObjectDetector] CreateEngine OK.");
  return true;
}

bool ObjectDetector::LoadEngine() {
  SPDLOG_DEBUG("[ObjectDetector] LoadEngine.");

  std::vector<char> engine_bin;
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (engine_file.good()) {
    engine_file.seekg(0, engine_file.end);
    engine_bin.resize(engine_file.tellg());
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(engine_bin.data(), engine_bin.size());
    engine_file.close();
  } else {
    SPDLOG_ERROR("[ObjectDetector] LoadEngine Fail. Could not open file.");
    return false;
  }

  auto runtime = UniquePtr<IRuntime>(createInferRuntime(logger));

  engine = std::shared_ptr<ICudaEngine>(
      runtime->deserializeCudaEngine(engine_bin.data(), engine_bin.size()),
      InferDeleter());

  if (!engine) {
    SPDLOG_ERROR("[ObjectDetector] LoadEngine Fail.");
    return false;
  }
  SPDLOG_DEBUG("[ObjectDetector] LoadEngine OK.");
  return true;
}

bool ObjectDetector::SaveEngine() {
  SPDLOG_ERROR("[ObjectDetector] SaveEngine.");

  if (engine) {
    auto engine_serialized = UniquePtr<IHostMemory>(engine->serialize());
    std::ofstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
      SPDLOG_ERROR("[ObjectDetector] SaveEngine Fail. Could not open file.");
      return false;
    }
    engine_file.write(reinterpret_cast<const char *>(engine_serialized->data()),
                      engine_serialized->size());

    SPDLOG_DEBUG("[ObjectDetector] SaveEngine OK.");
    return true;
  }
  SPDLOG_ERROR("[ObjectDetector] SaveEngine Fail. No engine.");
  return false;
}

bool ObjectDetector::CreateContex() {
  SPDLOG_DEBUG("[ObjectDetector] CreateContex.");
  auto context = UniquePtr<IExecutionContext>(engine->createExecutionContext());
  if (!context) {
    SPDLOG_ERROR("[ObjectDetector] CreateContex Fail.");
    return false;
  }
  SPDLOG_DEBUG("[ObjectDetector] CreateContex OK.");
  return true;
}

ObjectDetector::ObjectDetector(int index) {
  SPDLOG_DEBUG("[ObjectDetector] Constructing.");
  onnx_file_path = "./best.onnx";
  engine_path = onnx_file_path + ".engine";
  // camera.Open(0);
  if (!LoadEngine()) {
    CreateEngine();
    SaveEngine();
  }
  CreateContex();
  SPDLOG_DEBUG("[ObjectDetector] Constructed.");
}

ObjectDetector::~ObjectDetector() {
  SPDLOG_DEBUG("[ObjectDetector] Destructing.");
  // camera.Close();
  SPDLOG_DEBUG("[ObjectDetector] Destructed.");
}

bool ObjectDetector::Infer() {
  // CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
  //                       batchSize * 3 * INPUT_H * INPUT_W * sizeof(float),
  //                       cudaMemcpyHostToDevice, stream));
  // context.enqueue(batchSize, buffers, stream, nullptr);
  // CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
  //                       batchSize * OUTPUT_SIZE * sizeof(float),
  //                       cudaMemcpyDeviceToHost, stream));
  // cudaStreamSynchronize(stream);
  return true;
}

#endif
