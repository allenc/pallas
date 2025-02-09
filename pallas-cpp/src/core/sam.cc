#include "core/sam.h"

#include <opencv2/opencv.hpp>

namespace pallas {
SegmentAnything::SegmentAnything() {}
SegmentAnything::~SegmentAnything() {
    if (loadingModel) {
        return;
    }
    if (preprocessing) {
        return;
    }
    clearLoadModel();
    clearPreviousMasks();
}

bool SegmentAnything::clearLoadModel() {
    try {
        Ort::Session* pre = sessionEncoder.release();
        Ort::Session* sam = sessionDecoder.release();
        delete pre;
        delete sam;
        inputShapeEncoder.resize(0);
        outputShapeEncoder.resize(0);
        highResFeatures1Shape.resize(0);
        highResFeatures2Shape.resize(0);
        outputTensorValuesEncoder.resize(0);
        highResFeatures1.resize(0);
        highResFeatures2.resize(0);
    } catch (Ort::Exception& e) {
        return false;
    }
    return true;
}

void SegmentAnything::clearPreviousMasks() { previousMasks.resize(0); }

void SegmentAnything::resizePreviousMasks(int previousMaskIdx) {
    if (previousMasks.size() > previousMaskIdx + 1) {
        previousMasks.resize(previousMaskIdx + 1);
    }
}

void SegmentAnything::terminatePreprocessing() {
    runOptionsEncoder.SetTerminate();
    terminating = true;
}

bool modelExists(const std::string& modelPath) {
    std::ifstream f(modelPath);
    if (!f.good()) {
        return false;
    }
    return true;
}

bool SegmentAnything::loadModel(const std::string& encoderPath,
                                const std::string& decoderPath,
                                int threadsNumber, std::string device) {
    try {
        loadingStart();
        if (!clearLoadModel()) {
            loadingEnd();
            return false;
        }
        if (!modelExists(encoderPath) || !modelExists(decoderPath)) {
            loadingEnd();
            return false;
        }
        for (int i = 0; i < 2; i++) {
            auto& option = sessionOptions[i];
            option.SetIntraOpNumThreads(threadsNumber);
            option.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (device == "cpu") {
                continue;
            }
            if (device.substr(0, 5) == "cuda:") {
                int gpuDeviceId = std::stoi(device.substr(5));
                OrtCUDAProviderOptions options;
                options.device_id = gpuDeviceId;
                option.AppendExecutionProvider_CUDA(options);
            }
        }
        sessionEncoder = std::make_unique<Ort::Session>(
            env, encoderPath.c_str(), sessionOptions[0]);
        sessionDecoder = std::make_unique<Ort::Session>(
            env, decoderPath.c_str(), sessionOptions[1]);
        inputShapeEncoder = sessionEncoder->GetInputTypeInfo(0)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
        outputShapeEncoder = sessionEncoder->GetOutputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape();
        highResFeatures1Shape = sessionEncoder->GetOutputTypeInfo(1)
                                    .GetTensorTypeAndShapeInfo()
                                    .GetShape();
        highResFeatures2Shape = sessionEncoder->GetOutputTypeInfo(2)
                                    .GetTensorTypeAndShapeInfo()
                                    .GetShape();
    } catch (Ort::Exception& e) {
        loadingEnd();
        return false;
    }
    if (terminating) {
        loadingEnd();
        return false;
    }
    loadingEnd();
    return true;
}

void SegmentAnything::loadingStart() { loadingModel = true; }

void SegmentAnything::loadingEnd() {
    loadingModel = false;
    terminating = false;
}

cv::Size SegmentAnything::getInputSize() {
    return cv::Size((int)inputShapeEncoder[3], (int)inputShapeEncoder[2]);
}

std::vector<const char*> getInputNames(std::unique_ptr<Ort::Session>& session) {
    std::vector<const char*> inputNames;
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetInputCount(); ++i) {
        Ort::AllocatedStringPtr name_Ptr =
            session->GetInputNameAllocated(i, allocator);
        char* name = name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        char* name_new = new char[name_length];
        strncpy(name_new, name, name_length);
        inputNames.push_back(name_new);
    }
    return inputNames;
}

std::vector<const char*> getOutputNames(
    std::unique_ptr<Ort::Session>& session) {
    std::vector<const char*> outputNames;
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
        Ort::AllocatedStringPtr name_Ptr =
            session->GetOutputNameAllocated(i, allocator);
        char* name = name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        char* name_new = new char[name_length];
        strncpy(name_new, name, name_length);
        outputNames.push_back(name_new);
    }
    return outputNames;
}

bool SegmentAnything::preprocessImage(const cv::Mat& image) {
    try {
        preprocessingStart();
        if (image.size() !=
            cv::Size((int)inputShapeEncoder[3], (int)inputShapeEncoder[2])) {
            preprocessingEnd();
            return false;
        }
        if (image.channels() != 3) {
            preprocessingEnd();
            return false;
        }
        std::vector<float> inputTensorValuesFloat;
        inputTensorValuesFloat.resize(
            inputShapeEncoder[0] * inputShapeEncoder[1] * inputShapeEncoder[2] *
            inputShapeEncoder[3]);
        for (int i = 0; i < inputShapeEncoder[2]; i++) {
            for (int j = 0; j < inputShapeEncoder[3]; j++) {
                int64_t pos = i * inputShapeEncoder[3] + j;
                int64_t size = inputShapeEncoder[2] * inputShapeEncoder[3];
                inputTensorValuesFloat[pos + size * 0] =
                    (image.at<cv::Vec3b>(i, j)[2] / 255.0 - 0.485) / 0.229;
                inputTensorValuesFloat[pos + size * 1] =
                    (image.at<cv::Vec3b>(i, j)[1] / 255.0 - 0.456) / 0.224;
                inputTensorValuesFloat[pos + size * 2] =
                    (image.at<cv::Vec3b>(i, j)[0] / 255.0 - 0.406) / 0.225;
            }
        }
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValuesFloat.data(),
            inputTensorValuesFloat.size(), inputShapeEncoder.data(),
            inputShapeEncoder.size());
        outputTensorValuesEncoder =
            std::vector<float>(outputShapeEncoder[0] * outputShapeEncoder[1] *
                               outputShapeEncoder[2] * outputShapeEncoder[3]);
        std::vector<Ort::Value> outputTensors;
        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValuesEncoder.data(),
            outputTensorValuesEncoder.size(), outputShapeEncoder.data(),
            outputShapeEncoder.size()));
        highResFeatures1 = std::vector<float>(
            highResFeatures1Shape[0] * highResFeatures1Shape[1] *
            highResFeatures1Shape[2] * highResFeatures1Shape[3]);
        highResFeatures2 = std::vector<float>(
            highResFeatures2Shape[0] * highResFeatures2Shape[1] *
            highResFeatures2Shape[2] * highResFeatures2Shape[3]);
        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, highResFeatures1.data(), highResFeatures1.size(),
            highResFeatures1Shape.data(), highResFeatures1Shape.size()));
        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, highResFeatures2.data(), highResFeatures2.size(),
            highResFeatures2Shape.data(), highResFeatures2Shape.size()));

        if (terminating) {
            preprocessingEnd();
            return false;
        }
        runOptionsEncoder.UnsetTerminate();
        std::vector<const char*> inputNames = getInputNames(sessionEncoder);
        std::vector<const char*> outputNames = getOutputNames(sessionEncoder);
        sessionEncoder->Run(runOptionsEncoder, inputNames.data(), &inputTensor,
                            1, outputNames.data(), outputTensors.data(),
                            outputTensors.size());
        for (size_t i = 0; i < inputNames.size(); ++i) {
            delete[] inputNames[i];
        }
        for (size_t i = 0; i < outputNames.size(); ++i) {
            delete[] outputNames[i];
        }
    } catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        preprocessingEnd();
        return false;
    }
    preprocessingEnd();
    return true;
}

void SegmentAnything::preprocessingStart() { preprocessing = true; }

void SegmentAnything::preprocessingEnd() {
    preprocessing = false;
    terminating = false;
}

void SegmentAnything::setRectsLabels(const std::list<cv::Rect>& rects,
                                     std::vector<float>* inputPointValues,
                                     std::vector<float>* inputLabelValues) {
    for (auto& roi : rects) {
        (*inputPointValues).push_back((float)roi.x);
        (*inputPointValues).push_back((float)roi.y);
        (*inputLabelValues).push_back(2);
        (*inputPointValues).push_back((float)roi.br().x);
        (*inputPointValues).push_back((float)roi.br().y);
        (*inputLabelValues).push_back(3);
    }
}

void SegmentAnything::setPointsLabels(const std::list<cv::Point>& points,
                                      int label,
                                      std::vector<float>* inputPointValues,
                                      std::vector<float>* inputLabelValues) {
    for (auto& point : points) {
        (*inputPointValues).push_back((float)point.x);
        (*inputPointValues).push_back((float)point.y);
        (*inputLabelValues).push_back(label);
    }
}

void SegmentAnything::setDecorderTensorsEmbeddings(
    std::vector<Ort::Value>* inputTensors) {
    (*inputTensors)
        .push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, (float*)outputTensorValuesEncoder.data(),
            outputTensorValuesEncoder.size(), outputShapeEncoder.data(),
            outputShapeEncoder.size()));
    (*inputTensors)
        .push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, (float*)highResFeatures1.data(),
            highResFeatures1.size(), highResFeatures1Shape.data(),
            highResFeatures1Shape.size()));
    (*inputTensors)
        .push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, (float*)highResFeatures2.data(),
            highResFeatures2.size(), highResFeatures2Shape.data(),
            highResFeatures2Shape.size()));
}

void SegmentAnything::setDecorderTensorsPointsLabels(
    std::vector<float>& inputPointValues, std::vector<float>& inputLabelValues,
    int batchNum, int numPoints, std::vector<Ort::Value>* inputTensors) {
    std::vector<int64_t> inputPointShape = {batchNum, numPoints, 2};
    std::vector<int64_t> inputLabelShape = {batchNum, numPoints};
    (*inputTensors)
        .push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputPointValues.data(), 2 * numPoints * batchNum,
            inputPointShape.data(), inputPointShape.size()));
    (*inputTensors)
        .push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputLabelValues.data(), numPoints * batchNum,
            inputLabelShape.data(), inputLabelShape.size()));
}

cv::Mat SegmentAnything::setDecorderTensorsImageSize(
    std::vector<int64_t>& orig_im_size_values_int64,
    std::vector<float>& orig_im_size_values_float,
    std::vector<Ort::Value>* inputTensors) {
    std::vector<int64_t> origImSizeShape = {2};
    cv::Mat outputMask;
    (*inputTensors)
        .push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, orig_im_size_values_int64.data(), 2,
            origImSizeShape.data(), origImSizeShape.size()));
    outputMask =
        cv::Mat((int)orig_im_size_values_int64[0],
                (int)orig_im_size_values_int64[1], CV_8UC1, cv::Scalar(0));

    return outputMask;
}

void SegmentAnything::setDecorderTensorsMaskInput(
    const size_t maskInputSize, float* maskInputValues, float* hasMaskValues,
    std::vector<float>& previousMaskInputValues,
    std::vector<Ort::Value>* inputTensors) {
    std::vector<int64_t> maskInputShape = {1, 1, 256, 256},
                         hasMaskInputShape = {1};
    if (hasMaskValues[0] == 1) {
        (*inputTensors)
            .push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, previousMaskInputValues.data(), maskInputSize,
                maskInputShape.data(), maskInputShape.size()));
    } else {
        (*inputTensors)
            .push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, maskInputValues, maskInputSize,
                maskInputShape.data(), maskInputShape.size()));
    }
    (*inputTensors)
        .push_back(Ort::Value::CreateTensor<float>(memoryInfo, hasMaskValues, 1,
                                                   hasMaskInputShape.data(),
                                                   hasMaskInputShape.size()));
}

cv::Mat SegmentAnything::getMaskBatch(std::vector<float>& inputPointValues,
                                      std::vector<float>& inputLabelValues,
                                      int batchNum, const cv::Size& imageSize) {
    std::vector<Ort::Value> inputTensors;
    setDecorderTensorsEmbeddings(&inputTensors);
    int numPoints = (int)inputLabelValues.size() / batchNum;
    setDecorderTensorsPointsLabels(inputPointValues, inputLabelValues, batchNum,
                                   numPoints, &inputTensors);
    const size_t maskInputSize = 256 * 256;
    std::vector<float> previousMaskInputValues;
    float maskInputValues[maskInputSize];
    memset(maskInputValues, 0, sizeof(maskInputValues));
    float hasMaskValues[] = {0};
    setDecorderTensorsMaskInput(maskInputSize, maskInputValues, hasMaskValues,
                                previousMaskInputValues, &inputTensors);
    std::vector<int64_t> orig_im_size_values_int64 = {imageSize.height,
                                                      imageSize.width};
    std::vector<float> orig_im_size_values_float = {
        (float)inputShapeEncoder[2], (float)inputShapeEncoder[3]};
    cv::Mat outputMask = setDecorderTensorsImageSize(
        orig_im_size_values_int64, orig_im_size_values_float, &inputTensors);
    try {
        Ort::RunOptions runOptionsDecoder;
        std::vector<const char*> inputNames = getInputNames(sessionDecoder);
        std::vector<const char*> outputNames = getOutputNames(sessionDecoder);
        auto outputTensors = sessionDecoder->Run(
            runOptionsDecoder, inputNames.data(), inputTensors.data(),
            inputTensors.size(), outputNames.data(), outputNames.size());
        for (size_t i = 0; i < inputNames.size(); ++i) {
            delete[] inputNames[i];
        }
        for (size_t i = 0; i < outputNames.size(); ++i) {
            delete[] outputNames[i];
        }
        auto scoreShape =
            outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
        auto scoreValues = outputTensors[1].GetTensorMutableData<float>();
        auto maskValues = outputTensors[0].GetTensorMutableData<float>();
        int batchNum = (int)scoreShape[0];
        int scoreNum = (int)scoreShape[1];
        for (int k = 0; k < batchNum; k++) {
            float maxScore = 0;
            int maxScoreIdx = 0;
            int offsetScore = k * scoreNum;
            for (int i = 0; i < scoreNum; i++) {
                if (scoreValues[offsetScore + i] > maxScore) {
                    maxScore = scoreValues[offsetScore + i];
                    maxScoreIdx = i;
                }
            }
            int offsetMask = k * scoreNum * outputMask.rows * outputMask.cols +
                             maxScoreIdx * outputMask.rows * outputMask.cols;
            for (int i = 0; i < outputMask.rows; i++) {
                for (int j = 0; j < outputMask.cols; j++) {
                    if (maskValues[offsetMask + i * outputMask.cols + j] > 0) {
                        outputMask.at<uchar>(i, j) = 255;
                    }
                }
            }
        }
    } catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return outputMask;
    }
    return outputMask;
}

cv::Mat SegmentAnything::getMask(std::vector<float>& inputPointValues,
                                 std::vector<float>& inputLabelValues,
                                 const cv::Size& imageSize, int previousMaskIdx,
                                 bool isNextGetMask) {
    std::vector<Ort::Value> inputTensors;
    setDecorderTensorsEmbeddings(&inputTensors);
    int numPoints = (int)inputLabelValues.size();
    setDecorderTensorsPointsLabels(inputPointValues, inputLabelValues, 1,
                                   numPoints, &inputTensors);
    const size_t maskInputSize = 256 * 256;
    std::vector<float> previousMaskInputValues;
    resizePreviousMasks(previousMaskIdx);
    float maskInputValues[maskInputSize];
    memset(maskInputValues, 0, sizeof(maskInputValues));
    float hasMaskValues[] = {0};
    if (isNextGetMask) {
    } else if (previousMaskIdx >= 0) {
        hasMaskValues[0] = 1;
        previousMaskInputValues = previousMasks[previousMaskIdx];
    }
    setDecorderTensorsMaskInput(maskInputSize, maskInputValues, hasMaskValues,
                                previousMaskInputValues, &inputTensors);
    std::vector<int64_t> orig_im_size_values_int64 = {imageSize.height,
                                                      imageSize.width};
    std::vector<float> orig_im_size_values_float = {
        (float)inputShapeEncoder[2], (float)inputShapeEncoder[3]};
    cv::Mat outputMask = setDecorderTensorsImageSize(
        orig_im_size_values_int64, orig_im_size_values_float, &inputTensors);
    try {
        Ort::RunOptions runOptionsDecoder;
        std::vector<const char*> inputNames = getInputNames(sessionDecoder);
        std::vector<const char*> outputNames = getOutputNames(sessionDecoder);
        auto outputTensors = sessionDecoder->Run(
            runOptionsDecoder, inputNames.data(), inputTensors.data(),
            inputTensors.size(), outputNames.data(), outputNames.size());
        for (size_t i = 0; i < inputNames.size(); ++i) {
            delete[] inputNames[i];
        }
        for (size_t i = 0; i < outputNames.size(); ++i) {
            delete[] outputNames[i];
        }
        int maxScoreIdx = 0;
        float maxScore = 0;
        auto scoreShape =
            outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
        auto scoreValues = outputTensors[1].GetTensorMutableData<float>();
        int scoreNum = (int)scoreShape[1];
        for (int i = 0; i < scoreNum; i++) {
            if (scoreValues[i] > maxScore) {
                maxScore = scoreValues[i];
                maxScoreIdx = i;
            }
        }
        int offsetMask = maxScoreIdx * outputMask.rows * outputMask.cols;
        int offsetLowRes = maxScoreIdx * maskInputSize;
        auto maskValues = outputTensors[0].GetTensorMutableData<float>();
        for (int i = 0; i < outputMask.rows; i++) {
            for (int j = 0; j < outputMask.cols; j++) {
                outputMask.at<uchar>(i, j) =
                    maskValues[offsetMask + i * outputMask.cols + j] > 0 ? 255
                                                                         : 0;
            }
        }
        previousMaskInputValues = std::vector<float>(maskInputSize);
        auto low_res_logits = outputTensors[2].GetTensorMutableData<float>();
        for (int i = 0; i < maskInputSize; i++) {
            previousMaskInputValues[i] = low_res_logits[offsetLowRes + i];
        }
        previousMasks.push_back(previousMaskInputValues);
    } catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return outputMask;
    }
    return outputMask;
}
}  // namespace pallas
