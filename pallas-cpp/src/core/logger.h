#pragma once

#include <spdlog/spdlog.h>

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

void init_logging();
}  // namespace pallas
