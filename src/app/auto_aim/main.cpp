#include "armor_detector.hpp"
#include "camera.hpp"
#include "compensator.hpp"
#include "robot.hpp"

class AutoAim : private App {
 private:
  Robot robot("/dev/ttyTHS2");
  Camera cam(0, 640, 480);
  ArmorDetector detector("RMUL2021_Armor.json", game::Team::kBLUE);
  Compensator compensator;

 public:
  Dart() {
    SPDLOG_WARN("***** Starting Auto aiming system. *****");
    PrepareLogging("logs/auto_aim.log");

    /* 初始化设备 */
    do {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (robot.GetTeam() != game::Team::kUNKNOWN);
  }

  ~Dart() {
    /* 关闭设备 */

    SPDLOG_WARN("***** Shuted Down Auto aiming system. *****");
  }

  /* 运行的主程序 */
  void Run() {
    while (1) {
      cv::Mat frame = cam.GetFrame();
      if (frame.empty()) continue;
      auto armors = detector.Detect(frame);
      // target = predictor.Predict(armors, frame);
      // compensator.Apply(target, frame, robot.GetRotMat());
      // robot.Aim(target.GetAimEuler(), false);
      detector.VisualizeResult(frame, 10);
      cv::imshow("show", frame);
      cv::waitKey(1);
    }
  }
};

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  AutoAim auto_aim;
  auto_aim.Run();

  return EXIT_SUCCESS;
}
