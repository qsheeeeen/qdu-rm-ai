#include "camera.hpp"

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

/**
 * @brief 用于抓取图片帧的线程
 *
 */
void Camera::GrabThread(void) {
  SPDLOG_DEBUG("[Camera] [GrabThread] Started.");
  int err = MV_OK;
  memset(&raw_frame, 0, sizeof(MV_FRAME_OUT));
  while (grabing) {
    err = MV_CC_GetImageBuffer(camera_handle_, &raw_frame, 1000);
    if (err == MV_OK) {
      SPDLOG_DEBUG("[Camera] [GrabThread] FrameNum: {}.",
                   raw_frame.stFrameInfo.nFrameNum);
    } else {
      SPDLOG_ERROR("[Camera] [GrabThread] GetImageBuffer fail! err: {0:x}.",
                   err);
    }

    cv::Mat raw_mat(
        cv::Size(raw_frame.stFrameInfo.nWidth, raw_frame.stFrameInfo.nHeight),
        CV_8UC3, raw_frame.pBufAddr);

    std::lock_guard<std::mutex> lock(frame_stack_mutex_);
    frame_stack_.push_front(raw_mat.clone());

    if (nullptr != raw_frame.pBufAddr) {
      err = MV_CC_FreeImageBuffer(camera_handle_, &raw_frame);
      if (err != MV_OK) {
        SPDLOG_ERROR("[Camera] [GrabThread] FreeImageBuffer fail! err: {0:x}.",
                     err);
      }
    }
  }
  SPDLOG_DEBUG("[Camera] [GrabThread] Stoped.");
}

/**
 * @brief 相机初始化前的准备工作
 *
 */
void Camera::Prepare() {
  int err = MV_OK;
  std::string err_msg;
  SPDLOG_DEBUG("[Camera] Prepare.");

  std::memset(&mv_dev_list_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  err = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &mv_dev_list_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] EnumDevices fail! err: {0:x}.", err);
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

/**
 * @brief Construct a new Camera object
 *
 */
Camera::Camera() {
  SPDLOG_DEBUG("[Camera] Constructing.");
  Prepare();
  SPDLOG_DEBUG("[Camera] Constructed.");
}

/**
 * @brief Construct a new Camera object
 *
 * @param index 相机索引号
 * @param height 输出图像高度
 * @param width 输出图像宽度
 */
Camera::Camera(unsigned int index, unsigned int height, unsigned int width)
    : frame_h_(height), frame_w_(width) {
  SPDLOG_DEBUG("[Camera] Constructing.");
  Prepare();
  Open(index);
  SPDLOG_DEBUG("[Camera] Constructed.");
}

/**
 * @brief Destroy the Camera object
 *
 */
Camera::~Camera() {
  SPDLOG_DEBUG("[Camera] Destructing.");
  Close();
  SPDLOG_DEBUG("[Camera] Destructed.");
}

/**
 * @brief 设置相机参数
 *
 * @param height 输出图像高度
 * @param width 输出图像宽度
 */
void Camera::Setup(unsigned int height, unsigned int width) {
  frame_h_ = height;
  frame_w_ = width;

  // TODO: 配置相机输入输出
}

/**
 * @brief 打开相机设备
 *
 * @param index 相机索引号
 * @return int 状态代码
 */
int Camera::Open(unsigned int index) {
  int err = MV_OK;
  std::string err_msg;

  SPDLOG_DEBUG("[Camera] Open index: {}.", index);

  if (index >= mv_dev_list_.nDeviceNum) {
    SPDLOG_ERROR("[Camera] Intput index:{} >= nDeviceNum:{} !", index,
                 mv_dev_list_.nDeviceNum);
    return MV_E_UNKNOW;
  }

  err = MV_CC_CreateHandle(&camera_handle_, mv_dev_list_.pDeviceInfo[index]);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] CreateHandle fail! err: {0:x}.", err);
    return err;
  }

  err = MV_CC_OpenDevice(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] OpenDevice fail! err: {0:x}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", 0);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] TriggerMode fail! err: {0:x}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "PixelFormat",
                           PixelType_Gvsp_RGB8_Packed);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] PixelFormat fail! err: {0:x}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "AcquisitionMode", 2);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] AcquisitionMode fail! err: {0:x}.", err);
    return err;
  }

  MVCC_FLOATVALUE frame_rate;
  err = MV_CC_GetFloatValue(camera_handle_, "ResultingFrameRate", &frame_rate);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] ResultingFrameRate fail! err: {0:x}.", err);
    return err;
  } else {
    SPDLOG_INFO("[Camera] ResultingFrameRate: {}.", frame_rate.fCurValue);
  }

  err = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", 2);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] ExposureAuto fail! err: {0:x}.", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "GammaSelector", 2);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] GammaSelector fail! err: {0:x}.", err);
    return err;
  }

  err = MV_CC_SetBoolValue(camera_handle_, "GammaEnable", true);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] GammaEnable fail! err: {0:x}.", err);
    return err;
  }

  err = MV_CC_StartGrabbing(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] StartGrabbing fail! err: {0:x}.", err);
    return err;
  }
  return MV_OK;
}

/**
 * @brief Get the Frame object
 *
 * @return cv::Mat 拍摄的图像
 */
cv::Mat Camera::GetFrame() {
  cv::Mat frame;

  std::lock_guard<std::mutex> lock(frame_stack_mutex_);
  if (!frame_stack_.empty()) {
    frame = frame_stack_.front();
    frame_stack_.clear();
  } else {
    SPDLOG_ERROR("[Camera] Empty frame stack!");
  }
  return frame;
}

/**
 * @brief 关闭相机设备
 *
 * @return int 状态代码
 */
int Camera::Close() {
  int err = MV_OK;
  std::string err_msg;

  SPDLOG_DEBUG("[Camera] Close.");

  err = MV_CC_StopGrabbing(camera_handle_);
  SPDLOG_ERROR("[Camera] StopGrabbing fail! err:{0:x}.", err);

  err = MV_CC_CloseDevice(camera_handle_);
  SPDLOG_ERROR("[Camera] ClosDevice fail! err:{0:x}.", err);

  err = MV_CC_DestroyHandle(camera_handle_);
  SPDLOG_ERROR("[Camera] DestroyHandle fail! err:{0:x}.", err);
  SPDLOG_DEBUG("[Camera] Closed.");
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
