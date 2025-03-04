#include "inference_service.h"

#include <core/logger.h>
#include <core/timer.h>

#include <expected>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

#include "mat_queue_utils.h"

namespace pallas {

InferenceService::InferenceService(InferenceServiceConfig config)
    : Service(std::move(config.base)),
      config_{config.inference},
      yolo_{config_.yolo_path, config_.yolo_labels_path, config_.use_gpu},
      sam_{},
      queue_by_name_{} {
    LOGI("Initializing Inference with {} shared memory queues: {}",
         config_.shared_memory_names.size(),
         fmt::join(config_.shared_memory_names, ","));
}

bool InferenceService::start() {
    auto open_result =
        utils::open_verified_queues<Queue>(config_.shared_memory_names);
    if (!open_result) {
        return false;
    }
    queue_by_name_ = std::move(*open_result);

    // Initailize vision models: YOLO (ctor) and SAM2
    sam_.loadModel(config_.sam_encoder_path, config_.sam_decoder_path,
                   std::thread::hardware_concurrency(),
                   config_.use_gpu ? "gpu" : "cpu");

    return Service::start();
}

void InferenceService::stop() { Service::stop(); }

std::expected<void, std::string> InferenceService::tick() {
    LOGI("InferenceService::tick()");

    Timer timer{};

    for (const auto& [name, queue_ptr] : queue_by_name_) {
        if (!queue_ptr) {
            LOGE("Invalid queue for {}", name);
            continue;
        }

        cv::Mat frame;
        if (!queue_ptr->try_pop(frame)) {
            LOGW("Failed to pop frame.");
        }
        if (frame.empty()) {
            continue;
        }

        // Check if there are any people with YOLO; thresholds match Ultralytics
        // default
        const float confidence_threshold = 0.25f;
        const float iou_threshold = 0.45f;
        timer.start("yolo detect");
        const auto& detections =
            yolo_.detect(frame, confidence_threshold, iou_threshold);
        timer.log_ms("yolo detect");
        if (detections.empty()) {
            continue;
        }
        const int person_class_id = 0;
        for (const auto& detection : detections) {
            if (detection.class_id != person_class_id) {
                continue;
            }

            // Have a frame with a person, pass the result to SAM.
            // TODO: Get cv::Mat of just the person.
            const cv::Size sam_size = sam_.getInputSize();
            cv::resize(frame, frame, sam_size);
            timer.start("sam preprocess");
            sam_.preprocessImage(frame);
            timer.log_ms("sam preprocess");
        }
    }

    return std::expected<void, std::string>{};
}

}  // namespace pallas
