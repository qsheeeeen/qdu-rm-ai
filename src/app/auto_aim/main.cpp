#include <iostream>

#include "behavior.hpp"
#include "camera.hpp"
#include "robot.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "vision.hpp"

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto file_sink =
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/log.txt", true);

  spdlog::sinks_init_list sink_list = {console_sink, file_sink};

  spdlog::set_default_logger(
      std::make_shared<spdlog::logger>("console & file", sink_list));

  spdlog::flush_on(spdlog::level::debug);
  spdlog::set_level(spdlog::level::trace);

  spdlog::warn("***** Running Auto Aim. *****");

  Robot bot("/home/qs/virtual_robot");
  Camera cam(0);

  // Test only. TOOD: Remove.
  TestTree();
  TestOpenCV();
  TestVideoWrite();

  // Init behavior.
  // Run true tree.

  return 0;
}
