#include "camera_service.h"

#include <core/logger.h>

#include <expected>
#include <filesystem>
#include <iostream>
#include <thread>

namespace pallas {

CameraService::CameraService(CameraServiceConfig config)
    : Service(std::move(config.base)),
      shared_memory_name_{config.shared_memory_name},
      shared_memory_frame_capacity_{config.shared_memory_frame_capacity},
      queue_{nullptr} {
    LOGI(
        "Initializing CameraService with shared memory queue {} with {} frame "
        "count "
        "capacity.",
        shared_memory_name_, shared_memory_frame_capacity_);
}

bool CameraService::start() {
    // Cleanup any previously existing shared memory and reinitalize the buffer
    Queue::Close(shared_memory_name_);
    queue_ = std::make_unique<Queue>(
        Queue::Create(shared_memory_name_, shared_memory_frame_capacity_));

    // Open the webcam stream
    capture_ = cv::VideoCapture(0);
    if (!capture_.isOpened()) {
        LOGW("Failed to open camera on start.");
        return false;
    }
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::filesystem::create_directories("./camera_service");

    return Service::start();
}

void CameraService::stop() {
    capture_.release();
    Service::stop();
}

Result<void> CameraService::tick() {
    if (!capture_.isOpened()) {
        return std::unexpected("Failed to open camera on tick.");
    }

    cv::Mat frame;
    capture_ >> frame;

    if (frame.empty()) {
        return std::unexpected("Failed to capture non-empty frame on tick.");
    }

    // Write the frame to shared memory
    if (!queue_->try_push(frame)) {
        return std::unexpected("Failed to push frame on tick.");
    }

    if (true)  // Write frames as images
    {
        static std::atomic<int> frame_idx{0};
        std::string filename =
            "./camera_service/frame_" + std::to_string(frame_idx++) + ".png";
        cv::imwrite(filename, frame);
    }

    return Result<void>{};
}

}  // namespace pallas
