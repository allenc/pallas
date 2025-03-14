#pragma once

#include <core/service.h>
#include <vision/sam.h>
#include <vision/yolo.h>

#include <expected>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include "mat_queue.h"

namespace pallas {

struct InferenceConfig {
    bool use_gpu;                                  // GPU flag
    std::filesystem::path yolo_path;               // YOLO11 model
    std::filesystem::path yolo_labels_path;        // YOLO11 classes
    std::filesystem::path sam_encoder_path;        // SAM2 preprocess model
    std::filesystem::path sam_decoder_path;        // SAM2 postprocess model
    std::vector<std::string> shared_memory_names;  // Image frame queues
};

struct InferenceServiceConfig {
    ServiceConfig base;
    InferenceConfig inference;
};

class InferenceService : public Service {
   public:
    using Service::Service;

    InferenceService(InferenceServiceConfig config);
    bool start() override;
    void stop() override;

   protected:
    using Queue = MatQueue<2764800>;  // MacOS frame size

    std::expected<void, std::string> tick() override;

    InferenceConfig config_;
    YouOnlyLookOnce yolo_;
    SegmentAnything sam_;
    std::unordered_map<std::string, std::unique_ptr<Queue>> queue_by_name_;
};
}  // namespace pallas
