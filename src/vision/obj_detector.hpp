#pragma once

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>

#include <memory>
#include <string>

#include "camera.hpp"

class InferDeleter {
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
  using UniquePtr = std::unique_ptr<T, InferDeleter>;

 private:
  std::string input_tensor_name;
  std::string output_tensor_names;
  std::string onnx_file_path;
  std::string engine_path;

  bool use_fp16;
  bool use_int8;

  // Camera camera;

  TRTLogger logger;

  nvinfer1::Dims dim_in;
  nvinfer1::Dims dim_out;
  int num_class{0};

  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  bool ProcessInput();
  bool Preprocesse();

  bool CreateEngine();
  bool LoadEngine();
  bool SaveEngine();
  bool CreateContex();

 public:
  ObjectDetector(int index);
  ~ObjectDetector();
  bool Build();
  bool Infer();
};
