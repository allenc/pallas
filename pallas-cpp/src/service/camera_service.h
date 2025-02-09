#pragma once

#include <core/mat_queue.h>
#include <core/result.h>
#include <core/service.h>

#include <cstddef>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

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
    Result<void> tick() override;

   private:
    using Queue = MatQueue<2764800>;

    std::string shared_memory_name_;
    std::size_t shared_memory_frame_capacity_;
    cv::VideoCapture capture_;
    std::unique_ptr<Queue> queue_;
};
}  // namespace pallas
