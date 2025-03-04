#include <fstream>
#include <opencv2/imgproc.hpp>
#include <random>

#include "../core/logger.h"
#include "yolo.h"

namespace pallas {
namespace utils {

std::vector<std::string> getClassNames(const std::string& path) {
    std::vector<std::string> classNames;
    std::ifstream infile(path);

    if (infile) {
        std::string line;
        while (getline(infile, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            classNames.emplace_back(line);
        }
    } else {
        LOGE("ERROR: Failed to access class name path: {}", path);
    }
    return classNames;
}

size_t vectorProduct(const std::vector<int64_t>& vector) {
    return std::accumulate(vector.begin(), vector.end(), 1ull,
                           std::multiplies<size_t>());
}

void letterBox(const cv::Mat& image, cv::Mat& outImage,
               const cv::Size& newShape, const cv::Scalar& color, bool auto_,
               bool scaleFill, bool scaleUp, int stride) {
    float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                           static_cast<float>(newShape.width) / image.cols);

    if (!scaleUp) {
        ratio = std::min(ratio, 1.0f);
    }

    int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
    int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

    int dw = newShape.width - newUnpadW;
    int dh = newShape.height - newUnpadH;

    if (auto_) {
        dw = (dw % stride) / 2;
        dh = (dh % stride) / 2;
    } else if (scaleFill) {
        newUnpadW = newShape.width;
        newUnpadH = newShape.height;
        ratio = std::min(static_cast<float>(newShape.width) / image.cols,
                         static_cast<float>(newShape.height) / image.rows);
        dw = 0;
        dh = 0;
    } else {
        int padLeft = dw / 2;
        int padRight = dw - padLeft;
        int padTop = dh / 2;
        int padBottom = dh - padTop;

        if (image.cols != newUnpadW || image.rows != newUnpadH) {
            cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0,
                       cv::INTER_LINEAR);
        } else {
            outImage = image;
        }

        cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft,
                           padRight, cv::BORDER_CONSTANT, color);
        return;
    }

    if (image.cols != newUnpadW || image.rows != newUnpadH) {
        cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0,
                   cv::INTER_LINEAR);
    } else {
        outImage = image;
    }

    int padLeft = dw / 2;
    int padRight = dw - padLeft;
    int padTop = dh / 2;
    int padBottom = dh - padTop;

    cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight,
                       cv::BORDER_CONSTANT, color);
}

BoundingBox scaleCoords(const cv::Size& imageShape, BoundingBox coords,
                        const cv::Size& imageOriginalShape, bool p_Clip) {
    BoundingBox result;
    float gain = std::min(static_cast<float>(imageShape.height) /
                              static_cast<float>(imageOriginalShape.height),
                          static_cast<float>(imageShape.width) /
                              static_cast<float>(imageOriginalShape.width));

    int padX = static_cast<int>(std::round(
        (imageShape.width - imageOriginalShape.width * gain) / 2.0f));
    int padY = static_cast<int>(std::round(
        (imageShape.height - imageOriginalShape.height * gain) / 2.0f));

    result.center.x =
        static_cast<int>(std::round((coords.center.x - padX) / gain));
    result.center.y =
        static_cast<int>(std::round((coords.center.y - padY) / gain));
    result.width = static_cast<int>(std::round(coords.width / gain));
    result.height = static_cast<int>(std::round(coords.height / gain));

    if (p_Clip) {
        result.center.x = clamp(result.center.x, 0, imageOriginalShape.width);
        result.center.y = clamp(result.center.y, 0, imageOriginalShape.height);
        result.width =
            clamp(result.width, 0, imageOriginalShape.width - result.center.x);
        result.height = clamp(result.height, 0,
                              imageOriginalShape.height - result.center.y);
    }
    return result;
}

void NMSBoxes(const std::vector<BoundingBox>& boundingBoxes,
              const std::vector<float>& scores, float scoreThreshold,
              float nmsThreshold, std::vector<int>& indices) {
    indices.clear();

    const size_t numBoxes = boundingBoxes.size();
    if (numBoxes == 0) {
        return;
    }

    std::vector<int> sortedIndices;
    sortedIndices.reserve(numBoxes);
    for (size_t i = 0; i < numBoxes; ++i) {
        if (scores[i] >= scoreThreshold) {
            sortedIndices.push_back(static_cast<int>(i));
        }
    }

    if (sortedIndices.empty()) {
        return;
    }

    std::sort(
        sortedIndices.begin(), sortedIndices.end(),
        [&scores](int idx1, int idx2) { return scores[idx1] > scores[idx2]; });

    std::vector<float> areas(numBoxes, 0.0f);
    for (size_t i = 0; i < numBoxes; ++i) {
        areas[i] = boundingBoxes[i].width * boundingBoxes[i].height;
    }

    std::vector<bool> suppressed(numBoxes, false);

    for (size_t i = 0; i < sortedIndices.size(); ++i) {
        int currentIdx = sortedIndices[i];
        if (suppressed[currentIdx]) {
            continue;
        }

        indices.push_back(currentIdx);

        const BoundingBox& currentBox = boundingBoxes[currentIdx];
        const float x1_max = currentBox.center.x;
        const float y1_max = currentBox.center.y;
        const float x2_max = currentBox.center.x + currentBox.width;
        const float y2_max = currentBox.center.y + currentBox.height;
        const float area_current = areas[currentIdx];

        for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
            int compareIdx = sortedIndices[j];
            if (suppressed[compareIdx]) {
                continue;
            }

            const BoundingBox& compareBox = boundingBoxes[compareIdx];
            const float x1 =
                std::max(x1_max, static_cast<float>(compareBox.center.x));
            const float y1 =
                std::max(y1_max, static_cast<float>(compareBox.center.y));
            const float x2 = std::min(
                x2_max,
                static_cast<float>(compareBox.center.x + compareBox.width));
            const float y2 = std::min(
                y2_max,
                static_cast<float>(compareBox.center.y + compareBox.height));

            const float interWidth = x2 - x1;
            const float interHeight = y2 - y1;

            if (interWidth <= 0 || interHeight <= 0) {
                continue;
            }

            const float intersection = interWidth * interHeight;
            const float unionArea =
                area_current + areas[compareIdx] - intersection;
            const float iou =
                (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

            if (iou > nmsThreshold) {
                suppressed[compareIdx] = true;
            }
        }
    }
}

std::vector<cv::Scalar> generateColors(
    const std::vector<std::string>& classNames, int seed) {
    static std::unordered_map<size_t, std::vector<cv::Scalar>> colorCache;

    size_t hashKey = 0;
    for (const auto& name : classNames) {
        hashKey ^= std::hash<std::string>{}(name) + 0x9e3779b9 +
                   (hashKey << 6) + (hashKey >> 2);
    }

    auto it = colorCache.find(hashKey);
    if (it != colorCache.end()) {
        return it->second;
    }

    std::vector<cv::Scalar> colors;
    colors.reserve(classNames.size());

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni(0, 255);

    for (size_t i = 0; i < classNames.size(); ++i) {
        colors.emplace_back(cv::Scalar(uni(rng), uni(rng), uni(rng)));
    }

    colorCache.emplace(hashKey, colors);
    return colorCache[hashKey];
}

void drawBoundingBox(cv::Mat& image, const std::vector<Detection>& detections,
                     const std::vector<std::string>& classNames,
                     const std::vector<cv::Scalar>& colors) {
    for (const auto& detection : detections) {
        // if (detection.conf <= CONFIDENCE_THRESHOLD) continue;

        if (detection.class_id < 0 ||
            static_cast<size_t>(detection.class_id) >= classNames.size())
            continue;

        const cv::Scalar& color = colors[detection.class_id % colors.size()];

        cv::rectangle(image,
                      cv::Point(detection.box.center.x, detection.box.center.y),
                      cv::Point(detection.box.center.x + detection.box.width,
                                detection.box.center.y + detection.box.height),
                      color, 2, cv::LINE_AA);

        std::string label =
            classNames[detection.class_id] + ": " +
            std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = std::min(image.rows, image.cols) * 0.0008;
        const int thickness = std::max(
            1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));
        int baseline = 0;

        cv::Size textSize =
            cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

        int labelY = std::max(detection.box.center.y, textSize.height + 5);
        cv::Point labelTopLeft(detection.box.center.x,
                               labelY - textSize.height - 5);
        cv::Point labelBottomRight(detection.box.center.x + textSize.width + 5,
                                   labelY + baseline - 5);

        cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

        cv::putText(image, label,
                    cv::Point(detection.box.center.x + 2, labelY - 2), fontFace,
                    fontScale, cv::Scalar(255, 255, 255), thickness,
                    cv::LINE_AA);
    }
}

void drawBoundingBoxMask(cv::Mat& image,
                         const std::vector<Detection>& detections,
                         const std::vector<std::string>& classNames,
                         const std::vector<cv::Scalar>& classColors,
                         float maskAlpha) {
    if (image.empty()) {
        LOGE("ERROR: Empty image provided to drawBoundingBoxMask.");
        return;
    }

    const int imgHeight = image.rows;
    const int imgWidth = image.cols;

    const double fontSize = std::min(imgHeight, imgWidth) * 0.0006;
    const int textThickness =
        std::max(1, static_cast<int>(std::min(imgHeight, imgWidth) * 0.001));

    cv::Mat maskImage(image.size(), image.type(), cv::Scalar::all(0));

    std::vector<const Detection*> filteredDetections;
    for (const auto& detection : detections) {
        if (  // detection.conf > CONFIDENCE_THRESHOLD &&
            detection.class_id >= 0 &&
            static_cast<size_t>(detection.class_id) < classNames.size()) {
            filteredDetections.emplace_back(&detection);
        }
    }

    for (const auto* detection : filteredDetections) {
        cv::Rect box(detection->box.center.x, detection->box.center.y,
                     detection->box.width, detection->box.height);
        const cv::Scalar& color = classColors[detection->class_id];
        cv::rectangle(maskImage, box, color, cv::FILLED);
    }

    cv::addWeighted(maskImage, maskAlpha, image, 1.0f, 0, image);

    for (const auto* detection : filteredDetections) {
        cv::Rect box(detection->box.center.x, detection->box.center.y,
                     detection->box.width, detection->box.height);
        const cv::Scalar& color = classColors[detection->class_id];
        cv::rectangle(image, box, color, 2, cv::LINE_AA);

        std::string label =
            classNames[detection->class_id] + ": " +
            std::to_string(static_cast<int>(detection->confidence * 100)) + "%";
        int baseLine = 0;
        cv::Size labelSize =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontSize,
                            textThickness, &baseLine);

        int labelY = std::max(detection->box.center.y, labelSize.height + 5);
        cv::Point labelTopLeft(detection->box.center.x,
                               labelY - labelSize.height - 5);
        cv::Point labelBottomRight(
            detection->box.center.x + labelSize.width + 5,
            labelY + baseLine - 5);

        cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

        cv::putText(image, label,
                    cv::Point(detection->box.center.x + 2, labelY - 2),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize,
                    cv::Scalar(255, 255, 255), textThickness, cv::LINE_AA);
    }
}

}  // namespace utils
}  // namespace pallas
