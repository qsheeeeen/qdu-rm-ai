#include <iostream>

#include "buff_detector.hpp"
#include "opencv2/opencv.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      "logs/buff.log", true);

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

  SPDLOG_WARN("***** Running buff. *****");

  cv::VideoCapture cap("../../../../image/redbuff.avi");
  cv::Mat frame;
  BuffDetector buff_detector("../../../../runtime/RMUT2021_Buff.json",
                             game::Team::kRED);
  while (cap.isOpened()) {
    cap >> frame;
    buff_detector.Detect(frame);
    buff_detector.VisualizeResult(frame, 5);
    cv::imshow("win", frame);
    if (' ' == cv::waitKey(5)) {
      cv::waitKey(0);
    }
  }

  return EXIT_SUCCESS;
}
