#include "app.hpp"
#include "armor_detector.hpp"
#include "compensator.hpp"
#include "hik_camera.hpp"
#include "robot.hpp"

class AutoAim : private App {
 private:
  Robot robot;
  HikCamera cam;
  ArmorDetector detector;
  Compensator compensator;

 public:
  AutoAim(const std::string& log_path) : App(log_path) {
    SPDLOG_WARN("***** Setting Up Auto Aiming system. *****");

    /* 初始化设备 */
    robot.Init("/dev/ttyTHS2");
    cam.Open(0);
    cam.Setup(640, 480);
    detector.LoadParams("RMUL2021_Armor.json");

    do {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (robot.GetTeam() != game::Team::kUNKNOWN);
  }

  ~AutoAim() {
    /* 关闭设备 */

    SPDLOG_WARN("***** Shuted Down Auto Aiming system. *****");
  }

  /* 运行的主程序 */
  void Run() {
    SPDLOG_WARN("***** Running Auto Aiming system. *****");

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

  AutoAim auto_aim("logs/auto_aim.log");
  auto_aim.Run();

  return EXIT_SUCCESS;
}
