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

class Object {
 private:
  float x_center_, y_center_, width_, height_;

 public:
  Object(x_center, y_center, width, height);
  ~Object = default;
  cv::Point Center();
  cv:Rect Rect();
};

class Detector {
  template <typename T>
  using UniquePtr = std::unique_ptr<T, TRTDeleter>;

 private:
  std::string onnx_file_path_;
  std::string engine_path_;

  TRTLogger logger_;

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;

  float conf_thres_, iou_thres_;
  int num_classes_, agnostic_;

  std::vector<void *> bindings_;
  int idx_in_;
  int idx_out_;

  Camera camera_;

  bool Preprocesse();

  bool CreateEngine();
  bool LoadEngine();
  bool SaveEngine();
  bool CreateContex();
  bool InitMemory();
  bool NonMaxSuppression();

 public:
  Detector();
  ~Detector();
  bool TestInfer();
  bool Infer();
};
