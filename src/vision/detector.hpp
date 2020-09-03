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

struct BBox {
  float x_ctr;
  float y_ctr;
  float w;
  float h;
};

struct alignas(float) Detection {
  struct BBox bbox;
  float conf;  // bbox_conf * cls_conf
  float class_id;
};

class Detector {
  template <typename T>
  using UniquePtr = std::unique_ptr<T, TRTDeleter>;

 private:
  std::string onnx_file_path_;
  std::string engine_path_;

  TRTLogger logger_;

  UniquePtr<nvinfer1::ICudaEngine> engine_;
  UniquePtr<nvinfer1::IExecutionContext> context_;

  float conf_thres_, nms_thres_, neighbor_thresh_;
  int num_classes_, agnostic_;

  std::vector<void *> bindings_;
  std::vector<size_t> bingings_size_;
  int idx_in_;
  int idx_out_;

  Camera camera_;

  bool CreateEngine();
  bool LoadEngine();
  bool SaveEngine();
  bool CreateContex();
  bool InitMemory();

 public:
  Detector();
  ~Detector();
  bool TestInfer();
  std::vector<Detection> Infer();
};
