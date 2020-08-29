#pragma once

#include <thread>

#include "MvCameraControl.h"

class Camera {
 private:
  MV_CC_DEVICE_INFO *mv_dev_info_;
  MV_CC_DEVICE_INFO_LIST mv_dev_list_;
  MVCC_INTVALUE init_val_;
  void *camera_handle_ = nullptr;
  bool continue_capture_ = false;
  std::thread capture_thread_;

  void WorkThread();
  void PrintDeviceInfo();
  void Prepare();

 public:
  Camera();
  Camera(unsigned int index);
  ~Camera();

  void Open(unsigned int index);
  bool GetFrame(void *output);
  int Close();
};
