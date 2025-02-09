#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pallas {

struct BoundingBox {
    int x;
    int y;
    int width;
    int height;

    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}
};

struct Detection {
    BoundingBox box;
    float conf{};
    int classId{};

    std::string to_string() const;
};

std::ostream& operator<<(std::ostream& os, const Detection& detection);

namespace utils {

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type inline clamp(
    const T& value, const T& low, const T& high) {
    T validLow = low < high ? low : high;
    T validHigh = low < high ? high : low;
    return value < validLow ? validLow
                            : (value > validHigh ? validHigh : value);
}

std::vector<std::string> getClassNames(const std::string& path);
size_t vectorProduct(const std::vector<int64_t>& vector);

void letterBox(const cv::Mat& image, cv::Mat& outImage,
               const cv::Size& newShape,
               const cv::Scalar& color = cv::Scalar(114, 114, 114),
               bool auto_ = true, bool scaleFill = false, bool scaleUp = true,
               int stride = 32);

BoundingBox scaleCoords(const cv::Size& imageShape, BoundingBox coords,
                        const cv::Size& imageOriginalShape, bool p_Clip);

void NMSBoxes(const std::vector<BoundingBox>& boundingBoxes,
              const std::vector<float>& scores, float scoreThreshold,
              float nmsThreshold, std::vector<int>& indices);

std::vector<cv::Scalar> generateColors(
    const std::vector<std::string>& classNames, int seed = 42);

void drawBoundingBox(cv::Mat& image, const std::vector<Detection>& detections,
                     const std::vector<std::string>& classNames,
                     const std::vector<cv::Scalar>& colors);

void drawBoundingBoxMask(cv::Mat& image,
                         const std::vector<Detection>& detections,
                         const std::vector<std::string>& classNames,
                         const std::vector<cv::Scalar>& classColors,
                         float maskAlpha = 0.4f);

};  // namespace utils

class YouOnlyLookOnce {
   public:
    YouOnlyLookOnce(const std::string& modelPath, const std::string& labelsPath,
                    bool useGPU = false);

    std::vector<Detection> detect(const cv::Mat& image,
                                  float confThreshold = 0.4f,
                                  float iouThreshold = 0.45f);

    void drawBoundingBox(cv::Mat& image,
                         const std::vector<Detection>& detections) const;

    void drawBoundingBoxMask(cv::Mat& image,
                             const std::vector<Detection>& detections,
                             float maskAlpha = 0.4f) const;

    const std::vector<std::string>& classNames() const;

   private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};
    bool isDynamicInputShape{};
    cv::Size inputImageShape;

    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char*> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char*> outputNames;

    size_t numInputNodes, numOutputNodes;

    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors;

    cv::Mat preprocess(const cv::Mat& image, float*& blob,
                       std::vector<int64_t>& inputTensorShape);

    std::vector<Detection> postprocess(
        const cv::Size& originalImageSize, const cv::Size& resizedImageShape,
        const std::vector<Ort::Value>& outputTensors, float confThreshold,
        float iouThreshold);
};

}  // namespace pallas
