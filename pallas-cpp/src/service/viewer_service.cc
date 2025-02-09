#include "viewer_service.h"

#include <core/logger.h>
#include <core/mat_queue_utils.h>
#include <fmt/format.h>

#include <chrono>
#include <expected>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>

namespace pallas {

ViewerService::ViewerService(ViewerServiceConfig config)
    : Service(std::move(config.base)),
      shared_memory_names_{config.shared_memory_names},
      queue_by_name_{} {
    LOGI(
        "Initializing ViewerService with {} shared memory queues: "
        "capacity.",
        shared_memory_names_.size(), fmt::join(shared_memory_names_, ","));
}

bool ViewerService::start() {
    auto open_result = utils::open_verified_queues<Queue>(shared_memory_names_);
    if (!open_result) {
        return false;
    }
    queue_by_name_ = std::move(*open_result);

    std::filesystem::create_directories("./viewer_service");

    return Service::start();
}

void ViewerService::stop() {
    queue_by_name_.clear();

    Service::stop();
}

Result<void> ViewerService::tick() {
    LOGI("ViewerService::tick()");
    for (const auto& [name, queue_ptr] : queue_by_name_) {
        if (!queue_ptr) {
            return std::unexpected(
                fmt::format("Failed to get queue {} on tick", name));
        }

        cv::Mat frame;
        if (!queue_ptr->try_pop(frame)) {
            return std::unexpected("Failed to pop frame on tick.");
        }

        if (!frame.empty()) {
            // Write frames as images
            static int frame_idx{0};
            std::string filename = "./viewer_service/frame_" +
                                   std::to_string(frame_idx++) + ".png";
            cv::imwrite(filename, frame);
        }
    }

    return Result<void>{};
}

}  // namespace pallas
