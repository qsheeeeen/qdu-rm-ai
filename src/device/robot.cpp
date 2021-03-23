#include "robot.hpp"

#include "opencv2/core/quaternion.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kLIMIT = 0.04;

}  // namespace

void Robot::ThreadRecv() {
  SPDLOG_DEBUG("[ThreadRecv] Started.");

  Protocol_ID_t id;
  Protocol_UpPackageReferee_t ref;
  Protocol_UpPackageMCU_t robot;

  while (thread_continue) {
    serial_.Recv(&id, sizeof(id));

    if (AI_ID_REF == id) {
      serial_.Recv(&ref, sizeof(ref));

      if (crc16::CRC16_Verify((uint8_t *)&ref, sizeof(ref))) {
        mutex_ref_.lock();
        std::memcpy(&ref_, &(ref.data), sizeof(ref_));
        mutex_ref_.unlock();
      }
    } else if (AI_ID_MCU == id) {
      serial_.Recv(&robot, sizeof(robot));

      if (crc16::CRC16_Verify((uint8_t *)&robot, sizeof(robot))) {
        mutex_mcu_.lock();
        std::memcpy(&mcu_, &(robot.data), sizeof(mcu_));
        mutex_mcu_.unlock();
      }
    }
  }
  SPDLOG_DEBUG("[ThreadRecv] Stoped.");
}

void Robot::ThreadTrans() {
  SPDLOG_DEBUG("[ThreadTrans] Started.");

  Protocol_DownPackage_t command;

  while (thread_continue) {
    mutex_commandq_.lock();
    if (!commandq_.empty()) {
      command.data = commandq_.front();
      command.crc16 = crc16::CRC16_Calc((uint8_t *)&command.data,
                                        sizeof(command.data), UINT16_MAX);
      serial_.Trans((char *)&command, sizeof(command));
      commandq_.pop();
    }
    mutex_commandq_.unlock();
  }
  SPDLOG_DEBUG("[ThreadTrans] Stoped.");
}

Robot::Robot(const std::string &dev_path) {
  serial_.Open(dev_path);
  serial_.Config();
  if (!serial_.IsOpen()) {
    SPDLOG_ERROR("Can't open device.");
  }

  thread_continue = true;
  thread_recv_ = std::thread(&Robot::ThreadRecv, this);
  thread_trans_ = std::thread(&Robot::ThreadTrans, this);

  SPDLOG_TRACE("Constructed.");
}

Robot::~Robot() {
  serial_.Close();

  thread_continue = false;
  thread_recv_.join();
  thread_trans_.join();
  SPDLOG_TRACE("Destructed.");
}

game::Team Robot::GetTeam() {
  if (ref_.team == AI_TEAM_RED)
    return game::Team::kBLUE;
  else if (ref_.team == AI_TEAM_BLUE)
    return game::Team::kRED;
  return game::Team::kUNKNOWN;
}

cv::Mat Robot::GetRotMat() {
  cv::Quatf q(mcu_.quat.q0, mcu_.quat.q1, mcu_.quat.q2, mcu_.quat.q3);
  return cv::Mat(q.toRotMat3x3(), true);
}

void Robot::Aim(common::Euler aiming_eulr, bool auto_fire) {
  data_.gimbal.pit = aiming_eulr.pitch;
  data_.gimbal.rol = aiming_eulr.roll;
  data_.gimbal.yaw = aiming_eulr.yaw;

  cv::Quatd q(mcu_.quat.q0, mcu_.quat.q1, mcu_.quat.q2, mcu_.quat.q3);
  cv::Vec3d vec = q.toEulerAngles(cv::QuatEnum::EXT_XYZ);

  if (!auto_fire)
    data_.notice |= 0x00;
  else {
    if (fabs(vec[0] - aiming_eulr.pitch) >= kLIMIT)
      ;
    else if (fabs(vec[1] - aiming_eulr.roll) >= kLIMIT)
      ;
    else if (fabs(vec[2] - aiming_eulr.pitch) >= kLIMIT)
      ;
    else
      data_.notice |= AI_NOTICE_FIRE;
  }
}

void Robot::Move() {}
