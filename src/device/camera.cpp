#include "camera.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <thread>

#include "opencv2/imgproc.hpp"
#include "spdlog/spdlog.h"
// TMP
#include "opencv2/opencv.hpp"

void Camera::PrintDeviceInfo(MV_CC_DEVICE_INFO *mv_dev_info) {
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

void Camera::Prepare() {
  int err = MV_OK;
  std::string err_msg;
  SPDLOG_DEBUG("[Camera] Prepare.");

  std::memset(&mv_dev_list_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  err = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &mv_dev_list_);
  if (err != MV_OK) {
    err_msg = "[Camera] EnumDevices fail! err: " + std::to_string(err);
    SPDLOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  if (mv_dev_list_.nDeviceNum > 0) {
    for (unsigned int i = 0; i < mv_dev_list_.nDeviceNum; ++i) {
      SPDLOG_INFO("[Camera] Device {} slected.", i);
      MV_CC_DEVICE_INFO *dev_info = mv_dev_list_.pDeviceInfo[i];
      if (dev_info == nullptr) {
        SPDLOG_ERROR("[Camera] Error Reading dev_info");
        throw std::runtime_error("[Camera] Error Reading dev_info");
      } else
        PrintDeviceInfo(dev_info);
    }
  } else {
    SPDLOG_ERROR("[Camera] Find No Devices!");
    throw std::runtime_error("[Camera] Find No Devices!");
  }
}

Camera::Camera() {
  SPDLOG_DEBUG("[Camera] Constructing.");
  Prepare();
  SPDLOG_DEBUG("[Camera] Constructed.");
}

Camera::Camera(unsigned int out_h, unsigned int out_w)
    : out_h_(out_h), out_w_(out_w) {
  SPDLOG_DEBUG("[Camera] Constructing.");
  Prepare();
  SPDLOG_DEBUG("[Camera] Constructed.");
}

Camera::Camera(unsigned int index, unsigned int out_h, unsigned int out_w)
    : out_h_(out_h), out_w_(out_w) {
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

void Camera::Setup(unsigned int out_h, unsigned int out_w) {
  out_h_ = out_h;
  out_w_ = out_w;
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
    SPDLOG_ERROR("[Camera] SetTrigger fail! err: {}.", err);
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
    SPDLOG_ERROR("[Camera] AcquisitionMode fail! err: {}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "GammaSelector", 2);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] AcquisitionMode fail! err: {}.", err);
    return err;
  }

  err = MV_CC_SetBoolValue(camera_handle_, "GammaEnable", true);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] AcquisitionMode fail! err: {}.", err);
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
  int err = MV_OK;
  MV_FRAME_OUT frame_out = {};

  cv::Mat image;

  if (!MV_CC_IsDeviceConnected(camera_handle_)) {
    SPDLOG_ERROR("[Camera] Camera disconnected.");
    goto finally;
  }

  err = MV_CC_GetImageBuffer(camera_handle_, &frame_out, 10);
  if (err == MV_OK) {
    SPDLOG_DEBUG("[Camera] Get One Frame: Width:{}, Height:{}.",
                 frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight);

    SPDLOG_DEBUG("[Camera] FrameNum: {}.", frame_out.stFrameInfo.nFrameNum);

    cv::Mat raw(
        cv::Size(frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight),
        CV_8UC3, frame_out.pBufAddr);

    // TEST ONLY
    cv::cvtColor(raw, raw, cv::COLOR_RGB2BGR);
    cv::imwrite("./image/camera_raw.jpg", raw);

    const int offset_w = (raw.cols - raw.rows) / 2;
    const cv::Rect roi(offset_w, 0, raw.rows, raw.rows);
    cv::resize(raw(roi), image, cv::Size(out_h_, out_w_));

    // TEST ONLY
    cv::imwrite("./image/camera_output.jpg", image);

  } else {
    SPDLOG_ERROR("[Camera] GetImageBuffer fail! err:{}.", err);
  }

finally:
  if (NULL != frame_out.pBufAddr) {
    err = MV_CC_FreeImageBuffer(camera_handle_, &frame_out);
    if (err != MV_OK) {
      SPDLOG_ERROR("[Camera] FreeImageBuffer fail! err:{}.", err);
    }
  }
  return image;
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
