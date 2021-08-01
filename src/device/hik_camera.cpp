#include "hik_camera.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <thread>

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

/**
 * @brief 打印设备信息
 *
 * @param mv_dev_info 设备信息结构体
 */
static void PrintDeviceInfo(MV_CC_DEVICE_INFO *mv_dev_info) {
  if (nullptr == mv_dev_info) {
    SPDLOG_ERROR("The Pointer of mv_dev_info is nullptr!");
    return;
  }
  if (mv_dev_info->nTLayerType == MV_USB_DEVICE) {
    SPDLOG_INFO("UserDefinedName: {}.",
                mv_dev_info->SpecialInfo.stUsb3VInfo.chUserDefinedName);

    SPDLOG_INFO("Serial Number: {}.",
                mv_dev_info->SpecialInfo.stUsb3VInfo.chSerialNumber);

    SPDLOG_INFO("Device Number: {}.",
                mv_dev_info->SpecialInfo.stUsb3VInfo.nDeviceNumber);
  } else {
    SPDLOG_WARN("Not supported.");
  }
}

void HikCamera::GrabPrepare() { std::memset(&raw_frame, 0, sizeof(raw_frame)); }

void HikCamera::GrabLoop() {
  int err = MV_OK;
  err = MV_CC_GetImageBuffer(camera_handle_, &raw_frame, 1000);
  if (err == MV_OK) {
    SPDLOG_DEBUG("[GrabThread] FrameNum: {}.", raw_frame.stFrameInfo.nFrameNum);
  } else {
    SPDLOG_ERROR("[GrabThread] GetImageBuffer fail! err: {0:x}.", err);
  }

  cv::Mat raw_mat(
      cv::Size(raw_frame.stFrameInfo.nWidth, raw_frame.stFrameInfo.nHeight),
      CV_8UC3, raw_frame.pBufAddr);

  std::lock_guard<std::mutex> lock(frame_stack_mutex_);
  frame_stack_.push_front(raw_mat.clone());

  if (nullptr != raw_frame.pBufAddr) {
    if ((err = MV_CC_FreeImageBuffer(camera_handle_, &raw_frame)) != MV_OK) {
      SPDLOG_ERROR("[GrabThread] FreeImageBuffer fail! err: {0:x}.", err);
    }
  }
}

bool HikCamera::OpenPrepare(unsigned int index) {
  int err = MV_OK;

  SPDLOG_DEBUG("Open index: {}.", index);

  if (index >= mv_dev_list_.nDeviceNum) {
    SPDLOG_ERROR("Intput index:{} >= nDeviceNum:{} !", index,
                 mv_dev_list_.nDeviceNum);
    return false;
  }

  err = MV_CC_CreateHandle(&camera_handle_, mv_dev_list_.pDeviceInfo[index]);
  if (err != MV_OK) {
    SPDLOG_ERROR("CreateHandle fail! err: {0:x}.", err);
    return false;
  }

  if ((err = MV_CC_OpenDevice(camera_handle_)) != MV_OK) {
    SPDLOG_ERROR("OpenDevice fail! err: {0:x}.", err);
    return false;
  }

  if ((err = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", 0)) != MV_OK) {
    SPDLOG_ERROR("TriggerMode fail! err: {0:x}.", err);
    return false;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "PixelFormat",
                           PixelType_Gvsp_RGB8_Packed);
  if (err != MV_OK) {
    SPDLOG_ERROR("PixelFormat fail! err: {0:x}.", err);
    return false;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "AcquisitionMode", 2);
  if (err != MV_OK) {
    SPDLOG_ERROR("AcquisitionMode fail! err: {0:x}.", err);
    return false;
  }

  MVCC_FLOATVALUE frame_rate;
  err = MV_CC_GetFloatValue(camera_handle_, "ResultingFrameRate", &frame_rate);
  if (err != MV_OK) {
    SPDLOG_ERROR("ResultingFrameRate fail! err: {0:x}.", err);
    return false;
  } else {
    SPDLOG_INFO("ResultingFrameRate: {}.", frame_rate.fCurValue);
  }

  if ((err = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", 2)) != MV_OK) {
    SPDLOG_ERROR("ExposureAuto fail! err: {0:x}.", err);
    return false;
  }

  if ((err = MV_CC_SetEnumValue(camera_handle_, "GammaSelector", 2)) != MV_OK) {
    SPDLOG_ERROR("GammaSelector fail! err: {0:x}.", err);
    return false;
  }

  if ((err = MV_CC_SetBoolValue(camera_handle_, "GammaEnable", true)) !=
      MV_OK) {
    SPDLOG_ERROR("GammaEnable fail! err: {0:x}.", err);
    return false;
  }

  if ((err = MV_CC_StartGrabbing(camera_handle_)) != MV_OK) {
    SPDLOG_ERROR("StartGrabbing fail! err: {0:x}.", err);
    return false;
  }
  return true;
}

/**
 * @brief 相机初始化前的准备工作
 *
 */
void HikCamera::Prepare() {
  int err = MV_OK;
  SPDLOG_DEBUG("Prepare.");

  std::memset(&mv_dev_list_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  err = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &mv_dev_list_);
  if (err != MV_OK) {
    SPDLOG_ERROR("EnumDevices fail! err: {0:x}.", err);
  }

  if (mv_dev_list_.nDeviceNum > 0) {
    for (unsigned int i = 0; i < mv_dev_list_.nDeviceNum; ++i) {
      SPDLOG_INFO("Device {} slected.", i);
      MV_CC_DEVICE_INFO *dev_info = mv_dev_list_.pDeviceInfo[i];
      if (dev_info == nullptr) {
        SPDLOG_ERROR("Error Reading dev_info");
      } else
        PrintDeviceInfo(dev_info);
    }
  } else {
    SPDLOG_ERROR("Find No Devices!");
  }
}

/**
 * @brief Construct a new HikCamera object
 *
 */
HikCamera::HikCamera() {
  Prepare();
  SPDLOG_TRACE("Constructed.");
}

/**
 * @brief Construct a new HikCamera object
 *
 * @param index 相机索引号
 * @param height 输出图像高度
 * @param width 输出图像宽度
 */
HikCamera::HikCamera(unsigned int index, unsigned int height,
                     unsigned int width) {
  Prepare();
  Open(index);
  Setup(height, width);
  SPDLOG_TRACE("Constructed.");
}

/**
 * @brief Destroy the HikCamera object
 *
 */
HikCamera::~HikCamera() {
  Close();
  SPDLOG_TRACE("Destructed.");
}

/**
 * @brief 关闭相机设备
 *
 * @return int 状态代码
 */
int HikCamera::Close() {
  grabing = false;
  grab_thread_.join();

  int err = MV_OK;
  if ((err = MV_CC_StopGrabbing(camera_handle_)) != MV_OK) {
    SPDLOG_ERROR("StopGrabbing fail! err:{0:x}.", err);
    return err;
  }
  if ((err = MV_CC_CloseDevice(camera_handle_)) != MV_OK) {
    SPDLOG_ERROR("ClosDevice fail! err:{0:x}.", err);
    return err;
  }
  if ((err = MV_CC_DestroyHandle(camera_handle_)) != MV_OK) {
    SPDLOG_ERROR("DestroyHandle fail! err:{0:x}.", err);
    return err;
  }
  SPDLOG_DEBUG("Closed.");

  return MV_OK;
}

/**
 * @brief 相机标定
 *
 */
void Calibrate() {
  // TODO
  return;
}
