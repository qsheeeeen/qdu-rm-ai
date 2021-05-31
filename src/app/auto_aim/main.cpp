#include <iostream>

#include "armor_detector.hpp"
#include "camera.hpp"
#include "compensator.hpp"
#include "robot.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      "logs/auto_aim.log", true);

  spdlog::sinks_init_list sink_list = {console_sink, file_sink};

  spdlog::set_default_logger(
      std::make_shared<spdlog::logger>("default", sink_list));

#if (SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_DEBUG)
  spdlog::flush_on(spdlog::level::debug);
  spdlog::set_level(spdlog::level::debug);
#elif (SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_INFO)
  spdlog::flush_on(spdlog::level::info);
  spdlog::set_level(spdlog::level::info);
#endif

  SPDLOG_WARN("***** Running Auto Aim. *****");

  Robot robot("/dev/ttyTHS2");
  Camera cam(0, 640, 480);
  cv::Mat frame;

  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } while (robot.GetTeam() != game::Team::kUNKNOWN);

  ArmorDetector detector("RMUL2021_Armor.json", game::Team::kBLUE);
  Compensator compensator;

  while (true) {
    frame = cam.GetFrame();
    if (frame.empty()) continue;
    auto armors = detector.Detect(frame);
    // target = predictor.Predict(armors, frame);
    // compensator.Apply(target, frame, robot.GetRotMat());
    // robot.Aim(target.GetAimEuler(), false);
    detector.VisualizeResult(frame, 10);
    cv::imshow("show", frame);
    cv::waitKey(1);
  }

  return EXIT_SUCCESS;
}
