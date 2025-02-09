#include "service.h"

#include <iostream>

#include "logger.h"

namespace pallas {
Service::Service() : thread_{}, running_{false}, base_config_() {}

Service::Service(ServiceConfig config)
    : thread_{}, running_{false}, base_config_(std::move(config)) {}

Service::~Service() { stop(); }

bool Service::start() {
    if (running_.load()) {
        LOGI("Service [{}] is already running.", base_config_.name);
        return false;
    }

    LOGI("Starting service [{}].\n{}", base_config_.name,
         base_config_.to_string());

    running_.store(true);
    thread_ =
        std::thread([this, process_interval = std::chrono::milliseconds(
                               static_cast<long>(base_config_.interval_ms))]() {
            while (running_.load()) {
                const auto start_time = std::chrono::steady_clock::now();

                if (auto tick_result = tick(); !tick_result) {
                    LOGI("Service [{}] failed to tick: {}", base_config_.name,
                         tick_result.error());
                    continue;
                }

                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time);

                const auto sleep_duration = process_interval - elapsed;

                if (sleep_duration.count() < 0) {
                    LOGI("Service [{}] is lagging by {} ms.", base_config_.name,
                         std::abs(sleep_duration.count()));
                    continue;
                }

                LOGI("Service [{}] sleeping for {} ms.", base_config_.name,
                     sleep_duration.count());

                std::this_thread::sleep_for(sleep_duration);
            }
        });

    return true;
}

void Service::stop() {
    running_.store(false);
    if (thread_.joinable()) {
        thread_.join();
    }
}

std::string ServiceConfig::to_string() const {
    std::ostringstream ss;
    ss << this;

    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const ServiceConfig& config) {
    os << fmt::format("ScenarioConfig(name={}, port={}, interval_ms{:.4f})",
                      config.name, config.port, config.interval_ms);

    return os;
}
}  // namespace pallas
