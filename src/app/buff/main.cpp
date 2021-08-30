#include "app.hpp"
#include "buff_detector.hpp"
#include "compensator.hpp"
#include "hik_camera.hpp"
#include "opencv2/opencv.hpp"
#include "predictor.hpp"
#include "robot.hpp"

class BuffAim : private App {
 private:
  Robot robot;
  HikCamera cam;
  BuffDetector detector;
  Predictor predictor;
  Compensator compensator;

 public:
  BuffAim(const std::string& log_path) : App(log_path) {
    SPDLOG_WARN("***** Setting Up Buff Aiming system. *****");

    /* 初始化设备 */
    robot.Init("/dev/ttyTHS2");
    cam.Open(1);
    cam.Setup(640, 480);
    detector.LoadParams("RUMT2021_Buff.json");
    compensator.LoadCameraMat("MV-CA016-10UC-6mm.json");

    do {
      predictor.SetTime(robot.GetTime());
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while ((robot.GetTeam() != game::Team::kUNKNOWN) &&
             (predictor.GetTime() != 0));
  }

  ~BuffAim() {
    /* 关闭设备 */

    SPDLOG_WARN("***** Shuted Down Buff Aiming system. *****");
  }

  void Run() {
    while (1) {
      cv::Mat frame = cam.GetFrame();
      if (frame.empty()) {
        SPDLOG_ERROR("cam.GetFrame is null");
        continue;
      }
      predictor.SetBuff(detector.Detect(frame).back());
      predictor.Predict();

      detector.VisualizeResult(frame, 5);
      predictor.VisualizePrediction(frame, true);
      cv::imshow("win", frame);
      if (' ' == cv::waitKey(10)) {
        cv::waitKey(0);
      }
    }
  }
};

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  BuffAim buff_aim("logs/buff_aim.log");
  buff_aim.Run();

  return EXIT_SUCCESS;
}
