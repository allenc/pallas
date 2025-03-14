#pragma once

#include <core/service.h>

#include <cstddef>
#include <expected>
#include <memory>
#include <string>

#include "ps3.h"
#include "mat_queue.h"

namespace pallas {

struct PS3CameraServiceConfig {
    ServiceConfig base;
    std::string shared_memory_name;
    std::size_t shared_memory_frame_capacity;
    PS3EyeConfig camera_config;
};

class PS3CameraService : public Service {
   public:
    using Service::Service;

    PS3CameraService(PS3CameraServiceConfig config);
    bool start() override;
    void stop() override;

   protected:
    std::expected<void, std::string> tick() override;

   private:
    using Queue = MatQueue<76800>; // Same size as used in CameraService

    std::string shared_memory_name_;
    std::size_t shared_memory_frame_capacity_;
    PS3EyeConfig camera_config_;
    PS3EyeCamera camera_;
    std::unique_ptr<Queue> queue_;
};
}  // namespace pallas
