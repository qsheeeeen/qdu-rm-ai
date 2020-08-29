#include "obj_detector.hpp"

#include <stdexcept>

// #include "NvInfer.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include "spdlog/spdlog.h"

#if 0
using namespace nvinfer1;

void TRTLogger::log(Severity severity, const char *msg) {
  if (severity == kINTERNAL_ERROR) {
    spdlog::error(msg);
  } else if (severity == kERROR) {
    spdlog::error(msg);
  } else if (severity == kWARNING) {
    spdlog::error(msg);
  } else if (severity == kINFO) {
    spdlog::error(msg);
  } else if (severity == kVERBOSE) {
    spdlog::error(msg);
  }
}

bool ObjectDetector::ProcessInput(const BufferManager &buffers) {
  const int in_height = dim_in.d[2];
  const int in_width = dim_in.d[3];

  // Read a random digit file
  srand(unsigned(time(nullptr)));
  std::vector<uint8_t> fileData(in_height * in_width);
  mNumber = rand() % 10;
  readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs),
              fileData.data(), in_height, in_width);

  float *hostDataBuffer =
      static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
  for (int i = 0; i < in_height * in_width; i++) {
    hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
  }

  return true;
}

bool ObjectDetector::CreateEngine() {
  auto builder = UniquePtr<IBuilder>(createInferBuilder(logger));
  if (!builder) {
    SPDLOG_ERROR("[ObjectDetector] createInferBuilder Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[ObjectDetector] createInferBuilder OK.");

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

  dim_in = network->getInput(0)->getDimensions();
  dim_out = network->getOutput(0)->getDimensions();

  SPDLOG_INFO("dim_out: {d}, dim_in: {d}.", dim_in.nbDims, dim_out.nbDims)

  auto config = UniquePtr<IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    SPDLOG_ERROR("[ObjectDetector] createBuilderConfig Fail.");
    return false;
  } else
    SPDLOG_DEBUG("[ObjectDetector] createBuilderConfig OK.");

  auto parser = UniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger);
  if (!parser) {
    SPDLOG_ERROR("[ObjectDetector] createParser Fail.");
    return false;
  }else 
  SPDLOG_DEBUG("[ObjectDetector] createParser OK.");

  auto parsed = parser->parseFromFile(
      onnx_file_path.c_str(),
      static_cast<int>(logger.getReportableSeverity()));
  if (!parsed) {
    SPDLOG_ERROR("[ObjectDetector] parseFromFile Fail.");
    return false;
  }else 
  SPDLOG_DEBUG("[ObjectDetector] parseFromFile OK.");
  
  builder->setMaxBatchSize(1);
  config->setMaxWorkspaceSize(1 << 30);

  auto profile = builder->createOptimizationProfile();
  profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 256 , 256});
  profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 256 , 256});
  profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{32, 3, 256 , 256});    
  config->addOptimizationProfile(profile);
  
  if (use_fp16) config->setFlag(BuilderFlag::kFP16);

  if (use_int8) {
    config->setFlag(BuilderFlag::kINT8);
    setAllTensorScales(network.get(), 127.0f, 127.0f);
  }

  if (use_dla_core >= 0) {
    if (builder->getNbDLACores() == 0) {
      SPDLOG_ERROR(
          "Trying to use {d} DLA core on a platform that doesn't have any DLA "
          "cores",
          use_dla_core);
      throw std::runtime_error(
          "Error: use DLA core on a platfrom that doesn't have any DLA cores");
    }

    if (allow_gpu_fallback) {
      config->setFlag(BuilderFlag::kGPU_FALLBACK);
    }

    if (!builder->getInt8Mode() && !config->getFlag(BuilderFlag::kINT8)) {
      // User has not requested INT8 Mode.
      // By default run in FP16 mode. FP32 mode is not permitted.
      builder->setFp16Mode(builder->platformHasFastFp16() && true);
      config->setFlag(BuilderFlag::kFP16);
    }
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(use_dla_core);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
  }
  
  SPDLOG_INFO("[ObjectDetector] CreateEngine, please wait for a while...");

  engine = std::shared_ptr<ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config),
      InferDeleter());

  if (!engine) {
    SPDLOG_ERROR("[ObjectDetector] CreateEngine Fail.");
    return false;
  }
  SPDLOG_INFO("[ObjectDetector] CreateEngine OK.")
  return true;
}

bool ObjectDetector::LoadEngine() {
  std::string buffer = readBuffer(engine_path);
  if (buffer.size()) {
    auto runtime = UniquePtr<IRuntime>(createInferRuntime(logger));
    engine =
        runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
  }
  if (!engine) {
    SPDLOG_ERROR("[ObjectDetector] LoadEngine Fail.");
    return false;
  }
  SPDLOG_DEBUG("[ObjectDetector] LoadEngine OK.");
  return true;
}

bool ObjectDetector::SaveEngine() {
  if (engine) {
    auto engine_plan = UniquePtr<IHostMemory>(engine->serialize());
    writeBuffer(engine_plan->data(), engine_plan->size(), engine_path);
    SPDLOG_DEBUG("[ObjectDetector] SaveEngine OK.");
    return true;
  }
  SPDLOG_ERROR("[ObjectDetector] SaveEngine Fail.");
  return false;
}

bool ObjectDetector::CreateContex() {
  auto context = UniquePtr<IExecutionContext>(engine->createExecutionContext());
  if (!context) {
    SPDLOG_ERROR("[ObjectDetector] CreateContex Fail.");
    return false;
  }
  SPDLOG_DEBUG("[ObjectDetector] CreateContex OK.");
  return true;
}

ObjectDetector::ObjectDetector() {
  SPDLOG_DEBUG("[ObjectDetector] Creating.");
  engine_path = onnx_file_path + ".engine";
  camera.Open(0);
  if (!LoadEngine()) CreateEngine();
  CreateContex();
  SPDLOG_DEBUG("[ObjectDetector] Created.");
}

ObjectDetector::~ObjectDetector() {
  SPDLOG_DEBUG("[Robot] Destroying.");
  camera.Close();
  SPDLOG_DEBUG("[Robot] Destried.");
}

bool ObjectDetector::Infer() {
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                        batchSize * 3 * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(batchSize, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
                        batchSize * OUTPUT_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

#endif
