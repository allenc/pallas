#include "timer.h"

#include <fmt/format.h>

#include "core/logger.h"

namespace pallas {

Timer::Timer(const std::string& name) {
    if (!name.empty()) {
        start(name);
    }
}

void Timer::start(const std::string& name) {
    timers_[name] = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_ms(const std::string& name) const {
    auto timer_it = timers_.find(name);
    if (timer_it == timers_.end()) {
        LOGW("Timer {} not found, returning 0 ms elapsed.", name);
        return 0.0;
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time -
                                                     timer_it->second)
        .count();
}

void Timer::log_ms(const std::string& name, const std::string& message) const {
    const std::string log_postfix =
        message.empty() ? "." : fmt::format(": {}", message);
    LOGI("Timer {} took {}ms{}", name, elapsed_ms(name), log_postfix);
}

}  // namespace pallas
