#pragma once

// Define for Mongoose HTTP library
#define MG_ENABLE_OPENSSL 0

#include <core/service.h>
#include <mongoose.h>
#include <vision/yolo.h>

#include <atomic>
#include <cstddef>
#include <expected>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mat_queue.h"

namespace pallas {

struct StreamServiceConfig {
    ServiceConfig base;
    std::string shared_memory_name;
    uint16_t http_port;
    std::vector<std::string> camera_ids;
    bool use_person_detector = false;
    bool use_gpu = false;  // Use GPU for YOLO inference
    std::string active_detection_camera = "";  // Only run detection on this camera (empty = all)
    std::string yolo_model_path = "../assets/yolo11.onnx";
    std::string yolo_labels_path = "../assets/yolo11_labels.txt";
};

class StreamService : public Service {
   public:
    using Service::Service;

    StreamService(StreamServiceConfig config);
    bool start() override;
    void stop() override;
    
    // Set how often frames are processed (1 = every frame, 2 = every other frame, etc.)
    void setFrameProcessingRate(int every_n_frames);

    // Get a list of all available cameras
    std::vector<std::string> getCameraIds() const { return camera_ids_; }

    // Methods to access camera data
    void serveLatestFrame(const std::string& camera_id,
                          std::vector<uint8_t>& jpeg_buffer);
    nlohmann::json getCameraInfo(const std::string& camera_id);
    nlohmann::json getAllCamerasInfo();

   protected:
    std::expected<void, std::string> tick() override;

   private:
  // using Queue = MatQueue<2764800>; // Resolution for webcam
  using Queue = MatQueue<921600>; // Resolution for webcam  

    std::string shared_memory_name_;
    uint16_t http_port_;
    std::vector<std::string> camera_ids_;
    std::unordered_map<std::string, std::unique_ptr<Queue>> camera_queues_;
    std::unordered_map<std::string, cv::Mat> latest_frames_;

    // Mongoose HTTP server
    struct mg_mgr mgr_;
    std::atomic<bool> http_server_running_{false};
    std::thread http_server_thread_;

    // YOLO detection
    bool use_person_detector_;
    bool use_gpu_;  // Use GPU for YOLO inference
    std::string active_detection_camera_;  // Only run detection on this camera (empty = all)
    std::unique_ptr<YouOnlyLookOnce> yolo_;
    std::unordered_map<std::string, std::vector<Detection>> latest_detections_;
    
    // Frame processing control
    int frame_counter_{0};
    int process_every_n_frames_{3}; // Only process every 3rd frame for better performance

    // Mutex for thread safety
    std::mutex mutex_;

    // HTTP server event handler
    static void eventHandler(struct mg_connection* c, int ev, void* ev_data);

    // Helper methods for HTTP routing
    static void handleListCameras(struct mg_connection* c,
                                  StreamService* service);
    static void handleGetCameraInfo(struct mg_connection* c,
                                    const std::string& camera_id,
                                    StreamService* service);
    static void handleGetCameraFrame(struct mg_connection* c,
                                     const std::string& camera_id,
                                     StreamService* service);
    static void handleMjpegStream(struct mg_connection* c,
                                  const std::string& camera_id,
                                  StreamService* service);
    static void serveStaticFile(struct mg_connection* c,
                                const std::string& path);
};

}  // namespace pallas
