#include <iostream>

#include "behavior.hpp"
#include "hik_camera.hpp"
#include "robot.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      "logs/sentry.log", true);

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

  SPDLOG_WARN("***** Running Sentry. *****");

  Robot robot("/dev/tty");

  return EXIT_SUCCESS;
}
