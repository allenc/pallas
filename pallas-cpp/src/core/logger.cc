#include "logger.h"

#include <spdlog/sinks/stdout_color_sinks.h>

#include <memory>
#include <source_location>

namespace pallas {
void init_logging() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern(
        "[%Y-%m-%d %H:%M:%S.%e][%^%l%$[%s:%#\x1B[32m\x1B[0m %v");

    auto logger = std::make_shared<spdlog::logger>("main", console_sink);
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
    spdlog::flush_on(spdlog::level::err);
}
}  // namespace pallas
