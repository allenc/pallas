#pragma once

#include <expected>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <ostream>
#include <string>
#include <vector>

#include "geometry.h"
#include "yolo.h"

namespace pallas {

struct SegmentedObject {
    Point centroid;
    std::vector<Point> points;

    int class_id{0};
};

struct DetectionOptions {
    double confidence_threshold;
    double iou_threshold;
};

class DetectionContext {
   public:
    DetectionContext(const std::string& model_path,
                     const std::string& labels_path, const cv::Mat& image);
    std::expected<std::vector<Detection>, std::string> detect_people(
        const DetectionOptions& options);

   private:
    cv::Mat image_;
    std::unique_ptr<YouOnlyLookOnce> yolo_;
};

class GeometryContext {
   public:
    GeometryContext(const cv::Mat& image);
    std::expected<Segment, std::string> detect_segment(const Point& start,
                                                       const Point& end);

   private:
    cv::Mat image_;
};

}  // namespace pallas
