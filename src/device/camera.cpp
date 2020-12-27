#include "camera.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <thread>

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

static void PrintDeviceInfo(MV_CC_DEVICE_INFO *mv_dev_info) {
  if (nullptr == mv_dev_info) {
    SPDLOG_ERROR("[Camera] The Pointer of mv_dev_info is nullptr!");
    return;
  }
  if (mv_dev_info->nTLayerType == MV_USB_DEVICE) {
    SPDLOG_INFO("[Camera] UserDefinedName: {}.",
                mv_dev_info->SpecialInfo.stUsb3VInfo.chUserDefinedName);

    SPDLOG_INFO("[Camera] Serial Number: {}.",
                mv_dev_info->SpecialInfo.stUsb3VInfo.chSerialNumber);

    SPDLOG_INFO("[Camera] Device Number: {}.",
                mv_dev_info->SpecialInfo.stUsb3VInfo.nDeviceNumber);
  } else {
    SPDLOG_WARN("[Camera] Not support.");
  }
}

void Camera::GrabThread(void) {
  SPDLOG_DEBUG("[Camera] [GrabThread] Started.");
  int err = MV_OK;
  memset(&raw_frame, 0, sizeof(MV_FRAME_OUT));
  while (grabing) {
    err = MV_CC_GetImageBuffer(camera_handle_, &raw_frame, 1000);
    if (err == MV_OK) {
      SPDLOG_DEBUG("[Camera] FrameNum: {}.", raw_frame.stFrameInfo.nFrameNum);
    } else {
      SPDLOG_ERROR("[Camera] GetImageBuffer fail! err:{}.", err);
    }

    cv::Mat raw_mat(
        cv::Size(raw_frame.stFrameInfo.nWidth, raw_frame.stFrameInfo.nHeight),
        CV_8UC3, raw_frame.pBufAddr);

    frame_stack_.push(raw_mat.clone());

    if (nullptr != raw_frame.pBufAddr) {
      err = MV_CC_FreeImageBuffer(camera_handle_, &raw_frame);
      if (err != MV_OK) {
        SPDLOG_ERROR("[Camera] FreeImageBuffer fail! err:{}.", err);
      }
    }
  }
  SPDLOG_DEBUG("[Camera] [GrabThread] Stoped.");
}

void Camera::Prepare() {
  int err = MV_OK;
  std::string err_msg;
  SPDLOG_DEBUG("[Camera] Prepare.");

  std::memset(&mv_dev_list_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  err = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &mv_dev_list_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] EnumDevices fail! err: {}.", err);
  }

  if (mv_dev_list_.nDeviceNum > 0) {
    for (unsigned int i = 0; i < mv_dev_list_.nDeviceNum; ++i) {
      SPDLOG_INFO("[Camera] Device {} slected.", i);
      MV_CC_DEVICE_INFO *dev_info = mv_dev_list_.pDeviceInfo[i];
      if (dev_info == nullptr) {
        SPDLOG_ERROR("[Camera] Error Reading dev_info");
      } else
        PrintDeviceInfo(dev_info);
    }
  } else {
    SPDLOG_ERROR("[Camera] Find No Devices!");
  }
}

Camera::Camera() {
  SPDLOG_DEBUG("[Camera] Constructing.");
  Prepare();
  SPDLOG_DEBUG("[Camera] Constructed.");
}

Camera::Camera(unsigned int index, unsigned int height, unsigned int width)
    : frame_h_(height), frame_w_(width) {
  SPDLOG_DEBUG("[Camera] Constructing.");
  Prepare();
  Open(index);
  SPDLOG_DEBUG("[Camera] Constructed.");
}

Camera::~Camera() {
  SPDLOG_DEBUG("[Camera] Destructing.");
  Close();
  SPDLOG_DEBUG("[Camera] Destructed.");
}

void Camera::Setup(unsigned int height, unsigned int width) {
  frame_h_ = height;
  frame_w_ = width;

  // TODO: 配置相机输入输出
}

int Camera::Open(unsigned int index) {
  int err = MV_OK;
  std::string err_msg;

  SPDLOG_DEBUG("[Camera] Open index:{}.", index);

  if (index >= mv_dev_list_.nDeviceNum) {
    SPDLOG_ERROR("[Camera] Intput index:{} >= nDeviceNum:{} !", index,
                 mv_dev_list_.nDeviceNum);
    return MV_E_UNKNOW;
  }

  err = MV_CC_CreateHandle(&camera_handle_, mv_dev_list_.pDeviceInfo[index]);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] CreateHandle fail! err: {}.", err);
    return err;
  }

  err = MV_CC_OpenDevice(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] OpenDevice fail! err: {}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", 0);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] TriggerMode fail! err: {}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "PixelFormat",
                           PixelType_Gvsp_RGB8_Packed);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] PixelFormat fail! err: {}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "AcquisitionMode", 2);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] AcquisitionMode fail! err: {}.", err);
    return err;
  }

  MVCC_FLOATVALUE frame_rate;
  err = MV_CC_GetFloatValue(camera_handle_, "ResultingFrameRate", &frame_rate);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] ResultingFrameRate fail! err: {}.", err);
    return err;
  } else {
    SPDLOG_INFO("[Camera] ResultingFrameRate: {}.", frame_rate.fCurValue);
  }

  err = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", 2);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] ExposureAuto fail! err: {}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "GammaSelector", 2);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] GammaSelector fail! err: {}.", err);
    return err;
  }

  err = MV_CC_SetBoolValue(camera_handle_, "GammaEnable", true);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] GammaEnable fail! err: {}.", err);
    return err;
  }

  err = MV_CC_StartGrabbing(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] StartGrabbing fail! err: {}.", err);
    return err;
  }
  return MV_OK;
}

cv::Mat Camera::GetFrame() {
  cv::Mat frame;
  if (!frame_stack_.empty()) {
    frame = frame_stack_.top();
    frame_stack_.pop();
  } else {
    SPDLOG_ERROR("[Camera] Empty frame stack!");
  }
  return frame;
}

int Camera::Close() {
  int err = MV_OK;
  std::string err_msg;

  SPDLOG_DEBUG("[Camera] Close.");

  err = MV_CC_StopGrabbing(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] StopGrabbing fail! err:{}.", err);
    return err;
  }

  err = MV_CC_CloseDevice(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] ClosDevice fail! err:{}.", err);
    return err;
  }

  err = MV_CC_DestroyHandle(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] DestroyHandle fail! err:{}.", err);
    return err;
  }

  SPDLOG_DEBUG("[Camera] Closed.");
  return MV_OK;
}
