#pragma once
#include <chrono>
#include <string>
#include <unordered_map>

namespace pallas {
/**
 * Usage:
 *     Timer timer{};
 *     timer.start_ms("a");
 *     timer.start_ms("b");
 *     .. logic (1s elapsed) ..
 *     timer.log_ms("a", "messageA"); // Logs "a 1s: messageA"
 *     .. logic (1s elapsed) ..
 *     timer.log_ms("b", "messageB"); // logs "b 2s: messageB"
 */
class Timer {
   public:
    Timer();
    void start(const std::string& name);
    double elapsed_ms(const std::string& name) const;
    void log_ms(const std::string& name, const std::string& message = "") const;

   private:
    std::unordered_map<std::string,
                       std::chrono::high_resolution_clock::time_point>
        timers_;
};
}  // namespace pallas
