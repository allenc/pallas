#pragma once

#include <core/service.h>

#include <cstddef>
#include <expected>
#include <memory>
#include <opencv2/videoio.hpp>
#include <string>

#include "mat_queue.h"

namespace pallas {

struct CameraServiceConfig {
    ServiceConfig base;
    std::string shared_memory_name;
    std::size_t shared_memory_frame_capacity;
};

class CameraService : public Service {
   public:
    using Service::Service;

    CameraService(CameraServiceConfig config);
    bool start() override;
    void stop() override;

   protected:
    std::expected<void, std::string> tick() override;

   private:
    using Queue = MatQueue<2764800>;

    std::string shared_memory_name_;
    std::size_t shared_memory_frame_capacity_;
    cv::VideoCapture capture_;
    std::unique_ptr<Queue> queue_;
};
}  // namespace pallas
