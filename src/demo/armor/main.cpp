#include <iostream>

#include "armor_detector.hpp"
#include "camera.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace {

const std::string source_path = "../../../../image/2.avi";
const std::string output_path = "../../../../image/test_detect.avi";
const std::string params_path =
    "../../../../src/demo/armor/params_test_blue.json";

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
  spdlog::flush_on(spdlog::level::info);
  spdlog::set_level(spdlog::level::info);
#endif

  SPDLOG_WARN("***** Running Armor Detecting Demo. *****");

  cv::VideoCapture cap(source_path);
  if (!cap.isOpened()) {
    SPDLOG_ERROR("Can't open {}.", source_path);
    return -1;
  }

  cv::Size f_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
                  cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = cap.get(cv::CAP_PROP_FPS);
  const int delay = static_cast<int>(1000. / fps);
  const int codec = cap.get(cv::CAP_PROP_FOURCC);

  cv::VideoWriter writer(output_path, codec, fps, f_size);
  if (!writer.isOpened()) {
    SPDLOG_ERROR("Can't write to {}.", output_path);
    return -1;
  }

  ArmorDetector detector(params_path, game::Team::kBLUE);

  cv::Mat frame;
  while (cap.read(frame)) {
    detector.Detect(frame);
    detector.VisualizeResult(frame, 2);

    cv::imshow("vedio", frame);
    if (cv::waitKey(delay) == 'q') break;

    writer.write(frame);
  }
  return 0;
}
