#include "detection.h"

#include <fmt/format.h>

#include <sstream>

namespace pallas {

DetectionContext::DetectionContext(const std::string& model_path,
                                   const std::string& labels_path,
                                   const cv::Mat& image)
    : yolo_(std::make_unique<YouOnlyLookOnce>(model_path, labels_path, false)),
      image_{image} {}

std::expected<std::vector<Detection>, std::string>
DetectionContext::detect_people(const DetectionOptions& options) {
    const auto& class_names = yolo_->class_names();
    const std::size_t classes_count = class_names.size();
    if (classes_count == 0) {
        return std::unexpected(
            "Error finding people. Err=YOLO has empty class names.");
    }

    const auto& detections = yolo_->detect(image_, options.confidence_threshold,
                                           options.iou_threshold);
    std::vector<Detection> people_detections;
    people_detections.reserve(detections.size());
    const std::string person_id = "person";
    for (const auto& detection : detections) {
        if (detection.class_id >= classes_count) {
            continue;
        }
        if (class_names.at(detection.class_id) != person_id) {
            continue;
        }
        people_detections.push_back(detection);
    }

    return people_detections;
}

}  // namespace pallas
