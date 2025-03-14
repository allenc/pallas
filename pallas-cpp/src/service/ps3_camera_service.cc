#include "ps3_camera_service.h"

#include <core/logger.h>

#include <expected>
#include <filesystem>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>

namespace pallas {

PS3CameraService::PS3CameraService(PS3CameraServiceConfig config)
    : Service(std::move(config.base)),
      shared_memory_name_{config.shared_memory_name},
      shared_memory_frame_capacity_{config.shared_memory_frame_capacity},
      camera_config_{std::move(config.camera_config)},
      camera_{camera_config_},
      queue_{nullptr} {
    LOGI(
        "Initializing PS3CameraService with shared memory queue {} with {} frame "
        "count capacity.",
        shared_memory_name_, shared_memory_frame_capacity_);
    LOGI("PS3EyeCamera config: width={}, height={}, fps={}, device_id={}",
         camera_config_.width, camera_config_.height, camera_config_.fps, 
         camera_config_.device_id);
}

bool PS3CameraService::start() {
    // Cleanup any previously existing shared memory and reinitalize the buffer
    Queue::Close(shared_memory_name_);
    queue_ = std::make_unique<Queue>(
        Queue::Create(shared_memory_name_, shared_memory_frame_capacity_));

    // Open the PS3 Eye camera
    auto result = camera_.open();
    if (!result) {
        LOGW("Failed to open PS3 Eye camera on start: {}", result.error());
        return false;
    }

    std::filesystem::create_directories("./ps3_camera_service");

    return Service::start();
}

void PS3CameraService::stop() {
    camera_.close();
    Service::stop();
}

std::expected<void, std::string> PS3CameraService::tick() {
    if (!camera_.isOpen()) {
        return std::unexpected("PS3 Eye camera is not open on tick.");
    }

    auto frame_result = camera_.captureFrame();
    if (!frame_result) {
        return std::unexpected("Failed to capture frame from PS3 Eye camera: " + 
                              frame_result.error());
    }

    cv::Mat frame = frame_result.value();

    // Write the frame to shared memory
    if (!queue_->try_push(frame)) {
        return std::unexpected("Failed to push frame to shared memory on tick.");
    }

    if (false)  // Write frames as images (disabled by default)
    {
        static int frame_idx{0};
        std::string filename =
            "./ps3_camera_service/frame_" + std::to_string(frame_idx++) + ".png";
        cv::imwrite(filename, frame);
    }

    return std::expected<void, std::string>{};
}

}  // namespace pallas