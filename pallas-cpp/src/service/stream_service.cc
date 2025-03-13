#include "stream_service.h"

#include <core/logger.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <regex>
#include <thread>
#include <vector>

namespace pallas {

// Helper function to read a file into a string
static std::string readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Helper function to get MIME type from file extension
static std::string getMimeType(const std::string& path) {
    static std::unordered_map<std::string, std::string> mimeTypes = {
        {".html", "text/html"},
        {".css", "text/css"},
        {".js", "application/javascript"},
        {".json", "application/json"},
        {".png", "image/png"},
        {".jpg", "image/jpeg"},
        {".jpeg", "image/jpeg"},
        {".gif", "image/gif"},
        {".svg", "image/svg+xml"},
        {".ico", "image/x-icon"}};

    auto pos = path.find_last_of('.');
    if (pos != std::string::npos) {
        std::string ext = path.substr(pos);
        auto it = mimeTypes.find(ext);
        if (it != mimeTypes.end()) {
            return it->second;
        }
    }

    return "application/octet-stream";
}

StreamService::StreamService(StreamServiceConfig config)
    : Service(config.base),
      shared_memory_name_(config.shared_memory_name),
      http_port_(config.http_port),
      camera_ids_(config.camera_ids),
      use_person_detector_(config.use_person_detector),
      yolo_(nullptr) {
          
    if (use_person_detector_) {
        LOGI("Initializing YOLO person detector with model {} and labels {}", 
             config.yolo_model_path, config.yolo_labels_path);
        try {
            yolo_ = std::make_unique<YouOnlyLookOnce>(
                config.yolo_model_path, config.yolo_labels_path, false);
            LOGI("YOLO model loaded successfully");
        } catch (const std::exception& e) {
            LOGE("Failed to load YOLO model: {}", e.what());
            use_person_detector_ = false;
        }
    }
}

void StreamService::eventHandler(struct mg_connection* c, int ev,
                                 void* ev_data) {
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message* hm = (struct mg_http_message*)ev_data;
        StreamService* service = static_cast<StreamService*>(c->fn_data);

        // Convert URI to string for easier handling
        std::string uri(hm->uri.buf, hm->uri.len);

        // API endpoints only - no static file serving
        if (uri == "/api/cameras") {
            // List all cameras
            handleListCameras(c, service);
        } else if (uri.find("/api/cameras/") == 0 &&
                   uri.find("/stream") != std::string::npos) {
            // MJPEG streaming endpoint
            LOGD("MJPEG stream request: {}", uri);

            // Extract the camera ID
            size_t start_pos = strlen("/api/cameras/");
            size_t end_pos = uri.find("/stream", start_pos);
            std::string camera_id = uri.substr(start_pos, end_pos - start_pos);

            LOGD("Extracted camera ID for stream: {}", camera_id);
            handleMjpegStream(c, camera_id, service);
        } else if (uri.find("/api/cameras/") == 0 &&
                   uri.find("/frame") != std::string::npos) {
            // Simplified camera frame endpoint check
            LOGD("Camera frame request: {}", uri);

            // Extract the camera ID
            size_t start_pos = strlen("/api/cameras/");
            size_t end_pos = uri.find("/frame", start_pos);
            std::string camera_id = uri.substr(start_pos, end_pos - start_pos);

            LOGD("Extracted camera ID: {}", camera_id);
            handleGetCameraFrame(c, camera_id, service);
        } else if (std::regex_match(uri, std::regex(R"(/api/cameras/([\w\-]+))"))) {
            // Extract camera ID from URI (supporting alpha-numeric and hyphens)
            // using regex group to properly extract camera IDs with hyphens
            const char* start = uri.c_str() + 13;  // Skip "/api/cameras/"
            std::string camera_id(start, uri.c_str() + uri.length() - start);
            handleGetCameraInfo(c, camera_id, service);
        } else {
            // Return API description for unknown endpoints
            const char* headers =
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n";

            nlohmann::json api_info = {
                {"status", "running"},
                {"endpoints",
                 {"/api/cameras", "/api/cameras/{camera_id}",
                  "/api/cameras/{camera_id}/frame",
                  "/api/cameras/{camera_id}/stream"}},
                {"message", "Pallas Stream Service API"}};

            std::string json_str = api_info.dump(2);
            mg_http_reply(c, 200, headers, "%s", json_str.c_str());
        }
    }
}

void StreamService::handleListCameras(struct mg_connection* c,
                                      StreamService* service) {
    // Get cameras info as JSON
    nlohmann::json cameras_json = service->getAllCamerasInfo();
    std::string json_str = cameras_json.dump();

    // Send response
    mg_http_reply(
        c, 200,
        "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n",
        "%s", json_str.c_str());
}

void StreamService::handleGetCameraInfo(struct mg_connection* c,
                                        const std::string& camera_id,
                                        StreamService* service) {
    // Get camera info as JSON
    nlohmann::json camera_json = service->getCameraInfo(camera_id);
    std::string json_str = camera_json.dump();

    // Send response
    mg_http_reply(
        c, 200,
        "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n",
        "%s", json_str.c_str());
}

void StreamService::handleMjpegStream(struct mg_connection* c,
                                      const std::string& camera_id,
                                      StreamService* service) {
    // Mark this connection as a streamer
    c->is_resp =
        1;  // This tells Mongoose not to close the connection after response
    c->data[0] = 1;  // Use user data to mark this connection as MJPEG streamer

    // Store camera ID in the connection data for later use
    // We'll use a simple approach - store just a pointer to a newly allocated
    // string
    if (c->fn_data != service) {
        LOGE("Invalid service pointer in connection");
        mg_error(c, "Internal server error");
        return;
    }

    // Add the connection to the list of MJPEG streamers if not already there
    static std::unordered_map<mg_connection*, std::string> mjpeg_connections;
    mjpeg_connections[c] = camera_id;

    LOGI("Starting MJPEG stream for camera {}", camera_id);

    // Send MJPEG stream headers
    mg_printf(
        c, "%s",
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate, max-age=0\r\n"
        "Pragma: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n");

    // The first frame will be sent during the next tick cycle
    // We create a timer callback function that will be called by Mongoose
    // periodically to send new frames
    mg_timer_add(
        &service->mgr_, 25,  // 25ms timer (approx 40fps) for smoother streaming
        MG_TIMER_REPEAT | MG_TIMER_RUN_NOW,
        [](void* arg) {
            // Timer callback - send frame to all active MJPEG connections
            auto conn_map =
                static_cast<std::unordered_map<mg_connection*, std::string>*>(
                    arg);

            // Iterate through all connections
            for (auto it = conn_map->begin(); it != conn_map->end();) {
                auto conn = it->first;
                const auto& cam_id = it->second;

                // Get the service pointer
                StreamService* svc = static_cast<StreamService*>(conn->fn_data);

                // Skip connections that are closed or errored
                if (conn->is_closing || conn->is_draining) {
                    LOGI("Removing closed MJPEG connection for camera {}",
                         cam_id);
                    it = conn_map->erase(it);
                    continue;
                }

                // Create buffer for the response
                std::vector<uint8_t> jpeg_buffer;
                {
                    std::lock_guard<std::mutex> lock(svc->mutex_);
                    svc->serveLatestFrame(cam_id, jpeg_buffer);
                }

                if (!jpeg_buffer.empty()) {
                    // Send the MJPEG part header
                    mg_printf(conn,
                              "--mjpegstream\r\n"
                              "Content-Type: image/jpeg\r\n"
                              "Content-Length: %lu\r\n\r\n",
                              jpeg_buffer.size());

                    // Send the binary JPEG data
                    mg_send(conn, jpeg_buffer.data(), jpeg_buffer.size());
                    mg_send(conn, "\r\n", 2);  // End with CRLF
                }

                ++it;
            }
        },
        &mjpeg_connections);
}

void StreamService::handleGetCameraFrame(struct mg_connection* c,
                                         const std::string& camera_id,
                                         StreamService* service) {
    // Limit the rate of frame requests (no more than one per 25ms per
    // connection) - allows up to 40fps
    static std::unordered_map<void*, std::chrono::steady_clock::time_point>
        last_request_times;
    auto now = std::chrono::steady_clock::now();

    // Check if this client is requesting too quickly
    auto it = last_request_times.find(c);
    if (it != last_request_times.end()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now - it->second)
                           .count();
        if (elapsed < 25) {
            // Too many requests, respond with a 429 (Too Many Requests)
            LOGW(
                "Rate limit: connection requesting too quickly ({}ms), camera "
                "{}",
                elapsed, camera_id);
            mg_http_reply(c, 429, "Access-Control-Allow-Origin: *\r\n",
                          "Too many requests. Please wait at least 25ms "
                          "between requests.");
            return;
        }
    }

    // Update the last request time
    last_request_times[c] = now;

    // Create buffer for the response
    std::vector<uint8_t> jpeg_buffer;
    {
        std::lock_guard<std::mutex> lock(service->mutex_);
        service->serveLatestFrame(camera_id, jpeg_buffer);
    }

    if (!jpeg_buffer.empty()) {
        // Send the JPEG image
        LOGI("Sending frame for camera {}, size: {} bytes", camera_id,
             jpeg_buffer.size());

        // Send headers with additional cache control
        mg_printf(
            c, "%s",
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: image/jpeg\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n"  // Close connection after response
            "Content-Length: ");
        mg_printf(c, "%lu\r\n\r\n", jpeg_buffer.size());

        // Send binary data directly
        mg_send(c, jpeg_buffer.data(), jpeg_buffer.size());

        // Clean up connection-specific data for connections that are too old (5
        // minutes)
        auto clean_time = now - std::chrono::minutes(5);
        for (auto it = last_request_times.begin();
             it != last_request_times.end();) {
            if (it->second < clean_time) {
                it = last_request_times.erase(it);
            } else {
                ++it;
            }
        }
    } else {
        // Not found or error encoding
        LOGE("No valid frame available for camera {}", camera_id);
        mg_http_reply(c, 404, "Access-Control-Allow-Origin: *\r\n",
                      "Camera not found or no frame available");
    }
}

void StreamService::serveStaticFile(struct mg_connection* c,
                                    const std::string& path) {
    // Send a CORS headers for all responses
    const char* headers =
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n";

    // Get the current directory
    std::filesystem::path current_path = std::filesystem::current_path() / path;

    // Read the file
    std::string content = readFile(current_path.string());

    if (content.empty()) {
        // File not found
        LOGW("File: {}", current_path.string());
        mg_http_reply(c, 404, headers, "File not found");
        return;
    }

    // Get MIME type
    std::string mime_type = getMimeType(path);

    // Send response with CORS headers
    std::string response_headers =
        std::string("Content-Type: ") + mime_type + "\r\n" + headers;
    mg_http_reply(c, 200, response_headers.c_str(), "%s", content.c_str());
}

bool StreamService::start() {
    LOGI("StreamService starting");

    // Flag to generate test frames if no real camera is connected
    bool generate_test_frames = true;

    // Open all camera queues
    for (const auto& camera_id : camera_ids_) {
        auto queue = std::make_unique<Queue>(Queue::Open(camera_id));
        if (!queue->is_valid()) {
            LOGE(
                "Failed to open shared memory queue for camera {}, will "
                "generate test frames",
                camera_id);
            // Keep going, we'll generate test frames
        } else {
            LOGI("Successfully opened shared memory queue for camera {}",
                 camera_id);
            camera_queues_[camera_id] = std::move(queue);
            generate_test_frames = false;
        }
    }

    // Create a test frame right away
    if (generate_test_frames) {
        LOGI("Generating test frames since no camera queues are available");
        for (const auto& camera_id : camera_ids_) {
            // Create a basic colored frame with timestamp
            cv::Mat test_frame(480, 640, CV_8UC3,
                               cv::Scalar(0, 0, 255));  // Red frame

            // Add text
            cv::putText(test_frame, "TEST FRAME", cv::Point(200, 240),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5,
                        cv::Scalar(255, 255, 255), 2);
            cv::putText(test_frame, "Camera ID: " + camera_id,
                        cv::Point(180, 280), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(255, 255, 255), 2);
            cv::putText(test_frame,
                        "Time: " + std::to_string(std::time(nullptr)),
                        cv::Point(180, 320), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(255, 255, 255), 2);

            // Store in latest frames
            latest_frames_[camera_id] = test_frame.clone();
            LOGI("Generated test frame for camera {}", camera_id);
        }
    }

    try {
        // Initialize Mongoose event manager
        mg_mgr_init(&mgr_);

        // Create a listening connection
        std::string listen_addr =
            "http://0.0.0.0:" + std::to_string(http_port_);

        mg_connection* c =
            mg_http_listen(&mgr_, listen_addr.c_str(), eventHandler, this);
        c->fn_data = this;
        if (c == nullptr) {
            LOGE("Failed to create listening connection on {}", listen_addr);
            mg_mgr_free(&mgr_);
            return false;
        }

        LOGI("HTTP server listening on {}", listen_addr);

        // Start HTTP server in a separate thread
        http_server_running_ = true;
        http_server_thread_ = std::thread([this]() {
            while (http_server_running_) {
                // Process events
                mg_mgr_poll(&mgr_, 100);  // 100ms timeout
            }
        });

        LOGI("HTTP server thread started");
    } catch (const std::exception& e) {
        LOGE("Failed to start HTTP server: {}", e.what());
        return false;
    }

    return Service::start();
}

void StreamService::stop() {
    LOGI("StreamService stopping");

    // Stop HTTP server
    http_server_running_ = false;

    // Wait for the server thread to exit
    if (http_server_thread_.joinable()) {
        http_server_thread_.join();
    }

    // Free Mongoose event manager
    mg_mgr_free(&mgr_);

    // Clear camera queues
    camera_queues_.clear();

    Service::stop();
}

std::expected<void, std::string> StreamService::tick() {
    // Lock for thread safety when updating latest frames
    std::lock_guard<std::mutex> lock(mutex_);

    // Flag to track if we got frames from any queue
    bool any_frames_received = false;

    // Process frames from all camera queues
    for (auto& [camera_id, queue] : camera_queues_) {
        cv::Mat frame;
        if (queue->try_pop(frame)) {
            // Process frame and store the latest frame for each camera
            LOGI("New frame received from camera {}", camera_id);

            // Store the latest frame
            latest_frames_[camera_id] = frame.clone();
            any_frames_received = true;
            
            // Run person detection if enabled
            if (use_person_detector_ && yolo_) {
                try {
                    // Run YOLO detection with default thresholds
                    const float confidence_threshold = 0.25f;
                    const float iou_threshold = 0.45f;
                    
                    auto detections = yolo_->detect(frame, confidence_threshold, iou_threshold);
                    
                    // Store all detections for this camera
                    latest_detections_[camera_id] = detections;
                    
                    // Log detection results with more detail for debugging
                    if (!detections.empty()) {
                        LOGI("Detected {} objects in camera {}", detections.size(), camera_id);
                        
                        // Count person detections (class_id 0 is person in COCO dataset)
                        int person_count = 0;
                        for (const auto& detection : detections) {
                            // Log each detection for detailed debugging
                            std::string class_name = "unknown";
                            if (yolo_ && detection.class_id < static_cast<int>(yolo_->class_names().size())) {
                                class_name = yolo_->class_names()[detection.class_id];
                            }
                            
                            LOGI("  Detection: class_id={} ({}) conf={:.2f} at position [{},{}] size {}x{}",
                                detection.class_id, 
                                class_name,
                                detection.confidence,
                                detection.box.center.x, 
                                detection.box.center.y,
                                detection.box.width,
                                detection.box.height);
                                
                            if (detection.class_id == 0) { // Person class
                                person_count++;
                            }
                        }
                        
                        if (person_count > 0) {
                            LOGI("Detected {} people in camera {}", person_count, camera_id);
                        }
                    }
                } catch (const std::exception& e) {
                    LOGE("Error during person detection: {}", e.what());
                }
            }
        }
    }

    // If we didn't get any frames from queues, update test frames
    if (!any_frames_received && camera_queues_.empty()) {
        // Every 30 ticks (roughly once per second), update the test frames
        static int tick_counter = 0;
        if (tick_counter++ % 30 == 0) {
            for (const auto& camera_id : camera_ids_) {
                // Create a basic colored frame with timestamp (higher quality)
                cv::Mat test_frame(720, 1280, CV_8UC3,
                                   cv::Scalar(0, 0, 200));  // Deep blue frame with higher resolution

                // Add a larger, more visible header box
                cv::rectangle(test_frame, cv::Point(140, 140), cv::Point(1140, 580),
                            cv::Scalar(40, 40, 100), -1); // Filled rectangle
                cv::rectangle(test_frame, cv::Point(140, 140), cv::Point(1140, 580),
                            cv::Scalar(100, 100, 255), 5); // Border

                // Add text with larger, more readable fonts
                cv::putText(test_frame, "TEST FRAME", cv::Point(400, 280),
                            cv::FONT_HERSHEY_SIMPLEX, 2.5,
                            cv::Scalar(255, 255, 255), 3);
                cv::putText(test_frame, "Camera ID: " + camera_id,
                            cv::Point(350, 380), cv::FONT_HERSHEY_SIMPLEX, 1.8,
                            cv::Scalar(255, 255, 255), 2);

                // Add current time
                std::time_t now = std::time(nullptr);
                char time_str[100];
                std::strftime(time_str, sizeof(time_str), "%H:%M:%S",
                              std::localtime(&now));
                cv::putText(test_frame, "Time: " + std::string(time_str),
                            cv::Point(380, 480), cv::FONT_HERSHEY_SIMPLEX, 1.8,
                            cv::Scalar(255, 255, 255), 2);

                // Store in latest frames
                latest_frames_[camera_id] = test_frame.clone();
                LOGI("Updated test frame for camera {}", camera_id);
            }
        }
    }

    // Small sleep to avoid busy waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return {};
}

void StreamService::serveLatestFrame(const std::string& camera_id,
                                     std::vector<uint8_t>& jpeg_buffer) {
    auto it = latest_frames_.find(camera_id);
    if (it != latest_frames_.end()) {
        try {
            // Make a copy of the frame so we can draw on it
            cv::Mat frame_to_serve = it->second.clone();
            
            // Draw bounding boxes for detections if person detection is enabled
            if (use_person_detector_ && yolo_) {
                auto detection_it = latest_detections_.find(camera_id);
                if (detection_it != latest_detections_.end() && !detection_it->second.empty()) {
                    // Draw all detections on the frame
                    yolo_->drawBoundingBox(frame_to_serve, detection_it->second);
                    LOGI("Drew {} detection boxes on frame for camera {}", 
                         detection_it->second.size(), camera_id);
                }
            }
            
            // Resize the image to a higher resolution for better quality
            cv::Mat resized;
            cv::resize(frame_to_serve, resized, cv::Size(640, 480), 0, 0,
                       cv::INTER_LINEAR);

            // Convert to JPEG with higher quality for better visual appearance
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
            jpeg_buffer.clear();  // Make sure buffer is empty
            cv::imencode(".jpg", resized, jpeg_buffer, params);

            LOGI(
                "Encoded frame for camera {}, resized {}x{} → 640x480, JPEG "
                "size: {} bytes",
                camera_id, it->second.cols, it->second.rows,
                jpeg_buffer.size());
        } catch (const std::exception& e) {
            LOGE("Error encoding frame for camera {}: {}", camera_id, e.what());
            jpeg_buffer.clear();
        }
    } else {
        // Create a test pattern at higher resolution
        cv::Mat test_pattern(480, 640, CV_8UC3,
                             cv::Scalar(255, 0, 0));  // Blue background

        // Draw a white rectangle
        cv::rectangle(test_pattern, cv::Point(100, 100), cv::Point(540, 380),
                      cv::Scalar(255, 255, 255), 5);

        // Add text with larger font for better readability
        cv::putText(test_pattern, "No Camera Feed", cv::Point(160, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255),
                    2);
        cv::putText(test_pattern, "Camera ID: " + camera_id, cv::Point(160, 260),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255),
                    2);

        // Get current time
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%H:%M:%S",
                      std::localtime(&now));
        cv::putText(test_pattern, "Time: " + std::string(time_str),
                    cv::Point(160, 320), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(255, 255, 255), 2);

        // Encode to JPEG with higher quality
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
        try {
            jpeg_buffer.clear();  // Make sure buffer is empty
            cv::imencode(".jpg", test_pattern, jpeg_buffer, params);
            LOGI(
                "Created fallback test pattern (640x480) for camera {}, "
                "JPEG size: {} bytes",
                camera_id, jpeg_buffer.size());
        } catch (const std::exception& e) {
            LOGE("Error encoding test pattern for camera {}: {}", camera_id,
                 e.what());
            jpeg_buffer.clear();
        }
    }
}

nlohmann::json StreamService::getCameraInfo(const std::string& camera_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    nlohmann::json camera_info;

    bool online = camera_queues_.find(camera_id) != camera_queues_.end();
    camera_info["id"] = camera_id;
    camera_info["name"] = "Camera " + camera_id;
    camera_info["online"] = online;
    camera_info["location"] = "Location " + camera_id;

    // Add resolution if we have a frame
    auto it = latest_frames_.find(camera_id);
    if (it != latest_frames_.end()) {
        int width = it->second.cols;
        int height = it->second.rows;
        
        // Log the original resolution
        LOGI("Camera {} original resolution: {}x{}", camera_id, width, height);
        
        // Original resolution before resizing
        camera_info["resolution"] = {{"width", width}, {"height", height}};
        
        // Display resolution (after resizing)
        camera_info["display_resolution"] = {{"width", 640}, {"height", 480}};
    } else {
        // If no frame is available, use default values
        LOGI("No frame available for camera {}, using default resolution", camera_id);
        camera_info["resolution"] = {{"width", 1280}, {"height", 720}};
        camera_info["display_resolution"] = {{"width", 640}, {"height", 480}};
    }
    
    // Add detection information if available
    if (use_person_detector_ && yolo_) {
        auto detection_it = latest_detections_.find(camera_id);
        if (detection_it != latest_detections_.end() && !detection_it->second.empty()) {
            // Count detections by class
            std::unordered_map<int, int> class_counts;
            for (const auto& detection : detection_it->second) {
                class_counts[detection.class_id]++;
            }
            
            // Add detection information to the response
            nlohmann::json detections_json = nlohmann::json::array();
            for (const auto& detection : detection_it->second) {
                nlohmann::json detection_json;
                detection_json["class_id"] = detection.class_id;
                
                if (yolo_ && detection.class_id < static_cast<int>(yolo_->class_names().size())) {
                    detection_json["class_name"] = yolo_->class_names()[detection.class_id];
                } else {
                    detection_json["class_name"] = "unknown";
                }
                
                detection_json["confidence"] = detection.confidence;
                detection_json["box"] = {
                    {"center_x", detection.box.center.x},
                    {"center_y", detection.box.center.y},
                    {"width", detection.box.width},
                    {"height", detection.box.height}
                };
                detections_json.push_back(detection_json);
            }
            
            camera_info["detections"] = detections_json;
            camera_info["detection_counts"] = class_counts;
            
            // Quick summary of people detected
            camera_info["people_detected"] = class_counts[0];
        } else {
            // No detections for this camera
            camera_info["detections"] = nlohmann::json::array();
            camera_info["people_detected"] = 0;
        }
    }

    return camera_info;
}

nlohmann::json StreamService::getAllCamerasInfo() {
    std::lock_guard<std::mutex> lock(mutex_);

    nlohmann::json cameras_list = nlohmann::json::array();

    for (const auto& camera_id : camera_ids_) {
        nlohmann::json camera_info;
        bool online = camera_queues_.find(camera_id) != camera_queues_.end();
        camera_info["id"] = camera_id;
        camera_info["name"] = "Camera " + camera_id;
        camera_info["online"] = online;
        camera_info["location"] = "Location " + camera_id;

        cameras_list.push_back(camera_info);
    }

    return nlohmann::json{{"cameras", cameras_list}};
}

}  // namespace pallas
