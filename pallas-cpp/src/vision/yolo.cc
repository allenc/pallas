#include "yolo.h"

#include <opencv2/imgproc.hpp>

#include "../core/logger.h"

namespace pallas {

YouOnlyLookOnce::YouOnlyLookOnce(const std::string& modelPath,
                                 const std::string& labelsPath, bool useGPU) {
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    sessionOptions.SetIntraOpNumThreads(
        std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        // Get available providers and print them for diagnostics
        std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
        LOGI("Available ONNX Runtime providers:");
        for (const auto& provider : availableProviders) {
            LOGI(" - {}", provider);
        }
        
        auto cudaAvailable =
            std::find(availableProviders.begin(), availableProviders.end(),
                      "CUDAExecutionProvider");
                      
        if (useGPU && cudaAvailable != availableProviders.end()) {
            LOGI("Inference device: GPU");
            // Set CUDA device ID to 0
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            
            try {
                sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
                LOGI("Successfully added CUDA execution provider");
            } catch (const Ort::Exception& e) {
                LOGE("Failed to append CUDA execution provider: {}", e.what());
                LOGI("Falling back to CPU execution provider");
            }
        } else {
            if (useGPU) {
                LOGI("GPU not supported by ONNXRuntime build. Using CPU.");
            }
            LOGI("Inference device: CPU.");
        }
    } catch (const std::exception& e) {
        LOGE("Exception while setting up execution providers: {}", e.what());
        LOGI("Falling back to CPU execution provider");
    }

#ifdef _WIN32
    std::wstring w_modelPath(modelPath.begin(), modelPath.end());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShapeVec =
        inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape =
        (inputTensorShapeVec.size() >= 4) &&
        (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3] == -1);

    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    if (inputTensorShapeVec.size() >= 4) {
        inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]),
                                   static_cast<int>(inputTensorShapeVec[2]));
    } else {
        throw std::runtime_error("Invalid input tensor shape.");
    }

    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    classNames_ = utils::getClassNames(labelsPath);
    classColors = utils::generateColors(classNames_);

    LOGI("Model loaded with {} input nodes and {} output nodes.", numInputNodes,
         numOutputNodes);
}

cv::Mat YouOnlyLookOnce::preprocess(const cv::Mat& image, float*& blob,
                                    std::vector<int64_t>& inputTensorShape) {
    cv::Mat resizedImage;
    utils::letterBox(image, resizedImage, inputImageShape,
                     cv::Scalar(114, 114, 114), isDynamicInputShape, false,
                     true, 32);

    cv::Mat rgbImage;
    cv::cvtColor(resizedImage, rgbImage, cv::COLOR_BGR2RGB);

    rgbImage.convertTo(rgbImage, CV_32FC3, 1.0f / 255.0f);

    blob = new float[rgbImage.cols * rgbImage.rows * rgbImage.channels()];

    std::vector<cv::Mat> chw(rgbImage.channels());
    for (int i = 0; i < rgbImage.channels(); ++i) {
        chw[i] = cv::Mat(rgbImage.rows, rgbImage.cols, CV_32FC1,
                         blob + i * rgbImage.cols * rgbImage.rows);
    }
    cv::split(rgbImage, chw);

    return rgbImage;
}

std::vector<Detection> YouOnlyLookOnce::postprocess(
    const cv::Size& originalImageSize, const cv::Size& resizedImageShape,
    const std::vector<Ort::Value>& outputTensors, float confThreshold,
    float iouThreshold) {
    std::vector<Detection> detections;
    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    const std::vector<int64_t> outputShape =
        outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    const size_t num_features = outputShape[1];
    const size_t num_detections = outputShape[2];

    if (num_detections == 0) {
        return detections;
    }

    const int numClasses = static_cast<int>(num_features) - 4;
    if (numClasses <= 0) {
        return detections;
    }

    std::vector<BoundingBox> boxes;
    boxes.reserve(num_detections);
    std::vector<float> confs;
    confs.reserve(num_detections);
    std::vector<int> classIds;
    classIds.reserve(num_detections);
    std::vector<BoundingBox> nms_boxes;
    nms_boxes.reserve(num_detections);

    const float* ptr = rawOutput;

    for (size_t d = 0; d < num_detections; ++d) {
        float centerX = ptr[0 * num_detections + d];
        float centerY = ptr[1 * num_detections + d];
        float width = ptr[2 * num_detections + d];
        float height = ptr[3 * num_detections + d];

        int classId = -1;
        float maxScore = -FLT_MAX;
        for (int c = 0; c < numClasses; ++c) {
            const float score = ptr[d + (4 + c) * num_detections];
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore > confThreshold) {
            float left = centerX - width / 2.0f;
            float top = centerY - height / 2.0f;

            BoundingBox scaledBox = utils::scaleCoords(
                resizedImageShape,
                BoundingBox({static_cast<int>(left), static_cast<int>(top)},
                            width, height),
                originalImageSize, true);

            BoundingBox roundedBox;
            roundedBox.center.x = std::round(scaledBox.center.x);
            roundedBox.center.y = std::round(scaledBox.center.y);
            roundedBox.width = std::round(scaledBox.width);
            roundedBox.height = std::round(scaledBox.height);

            BoundingBox nmsBox = roundedBox;
            nmsBox.center.x += classId * 7680;
            nmsBox.center.y += classId * 7680;

            nms_boxes.emplace_back(nmsBox);
            boxes.emplace_back(roundedBox);
            confs.emplace_back(maxScore);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    utils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

    detections.reserve(indices.size());
    for (const int idx : indices) {
        detections.emplace_back(
            Detection{boxes[idx], classIds[idx], confs[idx]});
    }

    return detections;
}

std::vector<Detection> YouOnlyLookOnce::detect(const cv::Mat& image,
                                               float confThreshold,
                                               float iouThreshold) {
    if (image.empty()) {
        LOGW("Error: Empty image provided to detector");
        return {};
    }

    float* blobPtr = nullptr;
    std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height,
                                             inputImageShape.width};

    cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

    delete[] blobPtr;

    static Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size());

    std::vector<Ort::Value> outputTensors =
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                    numInputNodes, outputNames.data(), numOutputNodes);

    cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
                               static_cast<int>(inputTensorShape[2]));

    std::vector<Detection> detections =
        postprocess(image.size(), resizedImageShape, outputTensors,
                    confThreshold, iouThreshold);

    return detections;
}

void YouOnlyLookOnce::drawBoundingBox(
    cv::Mat& image, const std::vector<Detection>& detections) const {
    utils::drawBoundingBox(image, detections, classNames_, classColors);
}

void YouOnlyLookOnce::drawBoundingBoxMask(
    cv::Mat& image, const std::vector<Detection>& detections,
    float maskAlpha) const {
    utils::drawBoundingBoxMask(image, detections, classNames_, classColors,
                               maskAlpha);
}

const std::vector<std::string>& YouOnlyLookOnce::class_names() const {
    return classNames_;
}

std::string Detection::to_string() const {
    std::stringstream ss;
    ss << *this;

    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Detection& detection) {
    os << fmt::format(
        "Detection(box={{x={}, y={}, width={}, height={}}}, "
        "confidence={:.4f}, class_id={})",
        detection.box.center.x, detection.box.center.y, detection.box.width,
        detection.box.height, detection.confidence, detection.class_id);

    return os;
}

}  // namespace pallas
