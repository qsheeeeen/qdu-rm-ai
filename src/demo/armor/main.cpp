#include <iostream>

#include "armor_detector.hpp"
#include "camera.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace {
const std::string avi_path = "../../../image/2.avi";
const std::string output_video_path = "../../../image/test1.mp4";
const std::string output_video_path_gray = "../../../image/test_gray.mp4";
const std::string params_path = "../../../src/demo/armor/params_test_blue.json";
}  // namespace

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      "logs/radar.log", true);

  spdlog::sinks_init_list sink_list = {console_sink, file_sink};

  spdlog::set_default_logger(
      std::make_shared<spdlog::logger>("default", sink_list));

#if (SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_DEBUG)
  spdlog::flush_on(spdlog::level::debug);
  spdlog::set_level(spdlog::level::debug);
#elif (SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_INFO)
  spdlog::flush_on(spdlog::level::err);
  spdlog::set_level(spdlog::level::info);
#endif

  SPDLOG_WARN("***** Running Armor Detecting Demo. *****");

  cv::Mat frame, result, gray;
  cv::VideoCapture capture;

  capture.open(avi_path);
  if (!capture.isOpened()) {
    SPDLOG_ERROR("[MainTest] Can't Open Video {}.", avi_path);
    return -1;
  }

#ifdef x
  cv::VideoWriter outputVideo_gray, outputVideo;
  int code = outputVideo.fourcc('X', 'V', 'I', 'D');
  outputVideo.open(output_video_path, code, 30.0, cv::Size2d(640, 480));
  outputVideo_gray.open(output_video_path_gray, code, 30.0, cv::Size2d(640, 480));
  if (!outputVideo.isOpened()) {
    SPDLOG_ERROR("[MainTest] Can't Open Video {}.", output_video_path);
    return -1;
  }
#endif

  for (;;) {
    capture >> frame;
    if (frame.empty()) {
      SPDLOG_ERROR("[MainTest] Empty Frame.");
      return -2;
    } else {
      ArmorDetector detector(params_path, game::Team::kBLUE);
      std::vector<Armor> armors = detector.Detect(frame);
      if (armors.size() == 0) {
        SPDLOG_DEBUG("[Armors] Found Armor Is {}.", armors.size());
        result = frame;
      } else {
        result = frame.clone();
        cv::resize(result, result, cv::Size2d(640, 480));
        detector.VisualizeResult(result, true);
      }
      cv::threshold(result, gray, 200, 255, cv::THRESH_BINARY);
      cv::imshow("vedio", gray);
      if (cv::waitKey(33) == 'q') break;
    }

#ifdef x
    outputVideo.write(result);
    outputVideo_gray.write(gray);
#endif
  }
#ifdef x
  outputVideo.release();
#endif
  capture.release();
  return 0;
}
