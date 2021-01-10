#include <iostream>

#include "camera.hpp"
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
  spdlog::flush_on(spdlog::level::err);
  spdlog::set_level(spdlog::level::info);
#endif

  SPDLOG_WARN("***** Running Auto Aim. *****");

  Robot bot("/dev/tty");

  exit(0);
}
