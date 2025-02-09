#pragma once
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <source_location>

namespace pallas {
#define LOGT(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOGD(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOGI(...) SPDLOG_INFO(__VA_ARGS__)
#define LOGW(...) SPDLOG_WARN(__VA_ARGS__)
#define LOGE(...) SPDLOG_ERROR(__VA_ARGS__)
#define LOGC(...) SPDLOG_CRITICAL(__VA_ARGS__)
#define LOGA(condition, ...)          \
    if (!(condition)) {               \
        SPDLOG_CRITICAL(__VA_ARGS__); \
        assert(condition);            \
    }

inline void init_logging() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern(
        "[%Y-%m-%d %H:%M:%S.%e][%^%l%$][%s:%#\x1B[32m]\x1B[0m %v");

    auto logger = std::make_shared<spdlog::logger>("main", console_sink);
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
    spdlog::flush_on(spdlog::level::err);
}
}  // namespace pallas
