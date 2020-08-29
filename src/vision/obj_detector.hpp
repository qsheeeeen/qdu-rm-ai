#pragma once

#include <memory>
#include <string>

#include "camera.hpp"
#include "spdlog/spdlog.h"

class InferDeleter {
 public:
  template <typename T>
  void operator()(T *obj) const {
    if (obj) {
      SPDLOG_DEBUG("[InferDeleter] destroy.");
      obj->destroy();
    }
  }
};

class TRTLogger {
 public:
  TRTLogger() = default;
  ~TRTLogger() = default;
#if 0
  void log(Severity severity, const char* msg) {
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
#endif
};

class ObjectDetector {
  template <typename T>
  using UniquePtr = std::unique_ptr<T, InferDeleter>;

 private:
  std::string input_tensor_name;
  std::string output_tensor_names;
  std::string onnx_file_path;
  std::string engine_path;

  int use_dla_core;
  bool use_fp16;
  bool use_int8;
  bool allow_gpu_fallback;

  Camera camera;

  TRTLogger logger;

#if 0
  nvinfer1::Dims dim_in;
  nvinfer1::Dims dim_out;
  int num_class{0};

  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  bool ProcessInput(const samplesCommon::BufferManager &buffers);
  bool Preprocesse(const samplesCommon::BufferManager &buffers);
#endif

  bool CreateEngine();
  bool LoadEngine();
  bool SaveEngine();
  bool CreateContex();

 public:
  ObjectDetector();
  ~ObjectDetector();
  bool Build();
  bool Infer();
};
