#pragma once

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>

#include <memory>
#include <string>
#include <vector>

#include "camera.hpp"

class TRTDeleter {
 public:
  template <typename T>
  void operator()(T *obj) const;
};

class TRTLogger : public nvinfer1::ILogger {
 public:
  TRTLogger() = default;
  ~TRTLogger() = default;

  void log(Severity severity, const char *msg) override;
  int GetVerbosity();
};

class ObjectDetector {
  template <typename T>
  using UniquePtr = std::unique_ptr<T, TRTDeleter>;

 private:
  std::string input_tensor_name;
  std::string output_tensor_names;
  std::string onnx_file_path;
  std::string engine_path;

  // Camera camera;

  TRTLogger logger;

  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  std::shared_ptr<nvinfer1::IExecutionContext> context;

  std::vector<void *> bindings;
  int idx_in;
  int idx_out;

  bool ProcessInput();
  bool Preprocesse();

  bool CreateEngine();
  bool LoadEngine();
  bool SaveEngine();
  bool CreateContex();
  bool InitMemory();

 public:
  ObjectDetector();
  ~ObjectDetector();
  bool Infer();
};
