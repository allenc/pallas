#include <core/logger.h>
#include <core/timer.h>
#include <gtest/gtest.h>
#include <vision/yolo.h>

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pallas {
class YouOnlyLookOnceTests : public testing::Test {
   protected:
    void SetUp() override {}

    void TearDown() override {}
};
TEST_F(YouOnlyLookOnceTests, Beep) {
    // Precondition.
    const bool is_gpu = false;
    const std::filesystem::path assets_path = "../assets/";
    const std::filesystem::path yolo_path = assets_path / "yolo11.onnx";
    const std::filesystem::path yolo_labels_path =
        assets_path / "yolo11_labels.txt";
    YouOnlyLookOnce yolo(yolo_path, yolo_labels_path, is_gpu);

    // Under test.
    const float confidence_threshold =
        0.25f;  // Match Ultralytics default confidence threshold
    const float iou_threshold =
        0.45f;  // Match Ultralytics default IoU threshold
    const std::filesystem::path image_path = assets_path / "barty.jpg";
    auto image = cv::imread(image_path, cv::IMREAD_COLOR);
    Timer timer{"yolo detect"};
    const auto& detections =
        yolo.detect(image, confidence_threshold, iou_threshold);
    timer.log_ms("yolo detect");

    // Postcondition.
    const auto& class_names = yolo.class_names();
    LOGI("Got {} detections", detections.size());
    for (const auto& detection : detections) {
        LOGI("detection: {}: class: {}", detection.to_string(),
             class_names.at(detection.class_id));
    }
}
}  // namespace pallas

// #include <iostream>
// #include <vector>
// #include <thread>
// #include <atomic>

// #include <opencv2/highgui/highgui.hpp>

// #include "./ia/YOLO11.hpp"
// p
// int main()
// {
//     // Configuration parameters
//     const bool isGPU = false;
//     const std::string labelsPath = "./classes.txt";
//     const std::string modelPath = "./best.onnx";
//     const std::string videoSource = "./input.mov"; // your usb cam device
//     const std::string outputPath = "./output.mp4"; // path for output video
//     file

//     // Use the same default thresholds as Ultralytics CLI
//     const float confThreshold = 0.25f;  // Match Ultralytics default
//     confidence threshold const float iouThreshold = 0.45f;   // Match
//     Ultralytics default IoU threshold

//     std::cout << "Initializing YOLOv11 detector with model: " << modelPath <<
//     std::endl; std::cout << "Using confidence threshold: " << confThreshold
//     << ", IoU threshold: " << iouThreshold << std::endl;

//     // Initialize YOLO detector
//     YOLO11Detector detector(modelPath, labelsPath, isGPU);

//     // Open video capture
//     cv::VideoCapture cap;

//     // configure the best camera to iphone 11
//     cap.open(videoSource, cv::CAP_FFMPEG);
//     if (!cap.isOpened())
//     {
//         std::cerr << "Error: Could not open the camera!\n";
//         return -1;
//     }

//     // Get video properties for the writer
//     double fps = cap.get(cv::CAP_PROP_FPS);
//     int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
//     int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

//     // Initialize video writer
//     cv::VideoWriter videoWriter;
//     int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // MP4 codec

//     // Open the video writer
//     bool isWriterOpened = videoWriter.open(outputPath, fourcc, fps,
//     cv::Size(width, height), true); if (!isWriterOpened) {
//         std::cerr << "Error: Could not open video writer!\n";
//         return -1;
//     }

//     std::cout << "Recording output to: " << outputPath << std::endl;
//     std::cout << "Press 'q' to stop recording and exit" << std::endl;

//     int frame_count = 0;
//     double total_time = 0.0;

//     for (;;)
//     {
//         cv::Mat frame;
//         cap >> frame;
//         if (frame.empty())
//         {
//             std::cerr << "Error: Could not read a frame!\n";
//             break;
//         }

//         // Display the frame
//         cv::imshow("input", frame);

//         // Measure detection time
//         auto start_time = std::chrono::high_resolution_clock::now();

//         // Perform detection with the updated thresholds
//         std::vector<Detection> detections = detector.detect(frame,
//         confThreshold, iouThreshold);

//         auto end_time = std::chrono::high_resolution_clock::now();
//         auto duration =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
//         start_time).count(); total_time += duration; frame_count++;

//         // Create a copy for output with detections drawn
//         cv::Mat outputFrame = frame.clone();

//         // Draw bounding boxes and masks on the frame
//         detector.drawBoundingBoxMask(outputFrame, detections);

//         // Add FPS info
//         double fps = 1000.0 / (total_time / frame_count);
//         cv::putText(outputFrame, "FPS: " +
//         std::to_string(static_cast<int>(fps)),
//                    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
//                    cv::Scalar(0, 255, 0), 2);

//         // Write the processed frame to the output video
//         videoWriter.write(outputFrame);

//         // Display the frame
//         cv::imshow("Detections", outputFrame);

//         // Use a small delay and check for 'q' key press to quit
//         if (cv::waitKey(1) == 'q')
//         {
//             break;
//         }
//     }

//     // Release resources
//     cap.release();
//     videoWriter.release();
//     cv::destroyAllWindows();

//     std::cout << "Video processing completed. Output saved to: " <<
//     outputPath << std::endl; std::cout << "Average FPS: " << (1000.0 /
//     (total_time / frame_count)) << std::endl;

//     return 0;
// }
