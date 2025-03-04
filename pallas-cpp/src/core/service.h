#pragma once
#include <atomic>
#include <chrono>
#include <expected>
#include <sstream>
#include <string>
#include <thread>

namespace pallas {

struct ServiceConfig {
    std::string name;
    std::uint16_t port;
    double interval_ms;

    std::string to_string() const;
};

class Service {
   public:
    Service();
    Service(ServiceConfig config);
    virtual ~Service();
    virtual bool start();
    virtual void stop();

   protected:
    virtual std::expected<void, std::string> tick() = 0;

   private:
    std::thread thread_;
    std::atomic<bool> running_;
    ServiceConfig base_config_;
};

std::ostream& operator<<(std::ostream& os, const ServiceConfig& config);
}  // namespace pallas
