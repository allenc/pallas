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

// Cache for preprocessed JPEG frames to avoid redundant encoding
struct CachedFrame {
    std::vector<uint8_t> jpeg_data;
    std::chrono::steady_clock::time_point timestamp;
    int original_width;
    int original_height;
    bool has_detections;
};

// Static cache with a mutex to protect access
static std::mutex frame_cache_mutex;
static std::unordered_map<std::string, CachedFrame> frame_cache;
static constexpr int CACHE_TTL_MS = 32; // Cache for 32ms (~30fps)
static constexpr int JPEG_QUALITY_STREAMING = 85; // Better quality-to-size ratio
static constexpr int MAX_DISPLAY_WIDTH = 640; // Larger frames for better quality

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
      use_gpu_(config.use_gpu),
      active_detection_camera_(config.active_detection_camera),
      yolo_(nullptr) {
          
    if (use_person_detector_) {
        LOGI("Initializing YOLO person detector with model {} and labels {}", 
             config.yolo_model_path, config.yolo_labels_path);
        try {
            // Initialize YOLO with GPU acceleration if requested
            yolo_ = std::make_unique<YouOnlyLookOnce>(
                config.yolo_model_path, config.yolo_labels_path, use_gpu_);
                
            LOGI("YOLO model loaded successfully using {}", 
                 use_gpu_ ? "GPU acceleration" : "CPU only");
                 
            if (!active_detection_camera_.empty()) {
                LOGI("Active detection mode: only running detection on camera {}", 
                     active_detection_camera_);
            }
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
    // Validate the service pointer
    if (!service) {
        LOGE("Invalid service pointer in handleGetCameraInfo");
        mg_http_reply(c, 500, "Access-Control-Allow-Origin: *\r\n", 
                     "Internal server error");
        return;
    }
    
    // Check if camera ID exists
    bool camera_exists = false;
    {
        std::lock_guard<std::mutex> lock(service->mutex_);
        camera_exists = std::find(service->camera_ids_.begin(), 
                                 service->camera_ids_.end(), 
                                 camera_id) != service->camera_ids_.end();
    }
    
    if (!camera_exists) {
        LOGE("Camera ID not found: {}", camera_id);
        mg_http_reply(c, 404, "Access-Control-Allow-Origin: *\r\n", 
                     "Camera not found");
        return;
    }
    
    try {
        // Get camera info as JSON
        nlohmann::json camera_json = service->getCameraInfo(camera_id);
        std::string json_str = camera_json.dump();

        // Send response
        mg_http_reply(
            c, 200,
            "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n",
            "%s", json_str.c_str());
    } catch (const std::exception& e) {
        LOGE("Error generating camera info for {}: {}", camera_id, e.what());
        mg_http_reply(c, 500, "Access-Control-Allow-Origin: *\r\n", 
                     "Error generating camera info");
    }
}

void StreamService::handleMjpegStream(struct mg_connection* c,
                                      const std::string& camera_id,
                                      StreamService* service) {
    // Mark this connection as a streamer
    c->is_resp =
        1;  // This tells Mongoose not to close the connection after response
    c->data[0] = 1;  // Use user data to mark this connection as MJPEG streamer

    // Validate the service pointer and camera ID
    if (c->fn_data != service) {
        LOGE("Invalid service pointer in connection");
        mg_error(c, "Internal server error");
        return;
    }
    
    // Check if the camera ID is valid
    bool camera_exists = false;
    {
        std::lock_guard<std::mutex> lock(service->mutex_);
        camera_exists = std::find(service->camera_ids_.begin(), 
                                  service->camera_ids_.end(), 
                                  camera_id) != service->camera_ids_.end();
    }
    
    if (!camera_exists) {
        LOGE("Invalid camera ID: {}", camera_id);
        mg_error(c, "Camera not found");
        return;
    }

    // Add the connection to the list of MJPEG streamers if not already there
    // Use a mutex-protected map to avoid race conditions
    static std::mutex mjpeg_connections_mutex;
    static std::unordered_map<mg_connection*, std::string> mjpeg_connections;
    
    {
        std::lock_guard<std::mutex> lock(mjpeg_connections_mutex);
        mjpeg_connections[c] = camera_id;
    }

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
        &service->mgr_, 16,  // 16ms timer (approx 60fps) for smoother streaming
        MG_TIMER_REPEAT | MG_TIMER_RUN_NOW,
        [](void* arg) {
            // Timer callback - send frame to all active MJPEG connections
            auto* conn_map = static_cast<std::unordered_map<mg_connection*, std::string>*>(arg);
            if (!conn_map) return;  // Safety check
            
            // Use mutex to protect the connections map
            static std::mutex mjpeg_connections_mutex;
            
            // Group connections by camera to avoid redundant frame processing
            std::unordered_map<std::string, std::vector<mg_connection*>> connections_by_camera;
            std::vector<mg_connection*> expired_connections;
            
            // First, group connections by camera ID and identify expired ones
            {
                std::lock_guard<std::mutex> map_lock(mjpeg_connections_mutex);
                
                for (auto it = conn_map->begin(); it != conn_map->end(); ++it) {
                    auto* conn = it->first;
                    const auto& cam_id = it->second;
                    
                    // Skip connections that are closed or errored
                    if (!conn || conn->is_closing || conn->is_draining) {
                        expired_connections.push_back(conn);
                        continue;
                    }
                    
                    // Get the service pointer with validation
                    auto* svc = static_cast<StreamService*>(conn->fn_data);
                    if (!svc) {
                        expired_connections.push_back(conn);
                        continue;
                    }
                    
                    // Add to connections by camera
                    connections_by_camera[cam_id].push_back(conn);
                }
            }
            
            // Process each camera's frame once and send to all connections
            for (auto& [cam_id, connections] : connections_by_camera) {
                if (connections.empty()) continue;
                
                // Get the service pointer from the first connection
                auto* svc = static_cast<StreamService*>(connections[0]->fn_data);
                if (!svc) continue;
                
                try {
                    // Create buffer for the response
                    std::vector<uint8_t> jpeg_buffer;
                    bool camera_valid = false;
                    
                    {
                        std::lock_guard<std::mutex> lock(svc->mutex_);
                        // Check if camera still exists
                        camera_valid = std::find(svc->camera_ids_.begin(), 
                                               svc->camera_ids_.end(), cam_id) 
                                     != svc->camera_ids_.end();
                        
                        if (camera_valid) {
                            svc->serveLatestFrame(cam_id, jpeg_buffer);
                        }
                    }
                    
                    if (!camera_valid) {
                        // Camera no longer exists, mark all connections as expired
                        for (auto* conn : connections) {
                            expired_connections.push_back(conn);
                        }
                        continue;
                    }
                    
                    if (!jpeg_buffer.empty()) {
                        // Prepare the MJPEG part header once
                        std::string header = fmt::format(
                            "--mjpegstream\r\n"
                            "Content-Type: image/jpeg\r\n"
                            "Content-Length: {}\r\n\r\n",
                            jpeg_buffer.size());
                        
                        // Send to all connections for this camera
                        for (auto* conn : connections) {
                            if (!conn || conn->is_closing || conn->is_draining) {
                                expired_connections.push_back(conn);
                                continue;
                            }
                            
                            // Send header and data
                            mg_send(conn, header.data(), header.size());
                            mg_send(conn, jpeg_buffer.data(), jpeg_buffer.size());
                            mg_send(conn, "\r\n", 2);  // End with CRLF
                        }
                    }
                } catch (const std::exception& e) {
                    // On error, mark all connections for this camera as expired
                    static int error_log_counter = 0;
                    if (++error_log_counter % 10 == 0) {
                        LOGE("Error processing MJPEG stream for camera {}: {}", cam_id, e.what());
                    }
                    
                    for (auto* conn : connections) {
                        expired_connections.push_back(conn);
                    }
                }
            }
            
            // Remove expired connections
            if (!expired_connections.empty()) {
                std::lock_guard<std::mutex> map_lock(mjpeg_connections_mutex);
                for (auto* conn : expired_connections) {
                    conn_map->erase(conn);
                }
                
                // Only log if multiple connections were removed
                if (expired_connections.size() > 1) {
                    LOGI("Removed {} expired MJPEG connections", expired_connections.size());
                }
            }
        },
        &mjpeg_connections);
}

void StreamService::handleGetCameraFrame(struct mg_connection* c,
                                         const std::string& camera_id,
                                         StreamService* service) {
    // Validate service pointer first
    if (!service) {
        LOGE("Invalid service pointer in handleGetCameraFrame");
        mg_http_reply(c, 500, "Access-Control-Allow-Origin: *\r\n", 
                     "Internal server error");
        return;
    }
    
    // Check if the camera ID is valid
    bool camera_exists = false;
    {
        std::lock_guard<std::mutex> lock(service->mutex_);
        camera_exists = std::find(service->camera_ids_.begin(), 
                                  service->camera_ids_.end(), 
                                  camera_id) != service->camera_ids_.end();
    }
    
    if (!camera_exists) {
        LOGE("Invalid camera ID in frame request: {}", camera_id);
        mg_http_reply(c, 404, "Access-Control-Allow-Origin: *\r\n", 
                     "Camera not found");
        return;
    }
    
    // Limit the rate of frame requests (no more than one per 25ms per
    // connection) - allows up to 40fps
    static std::mutex request_times_mutex;
    static std::unordered_map<void*, std::chrono::steady_clock::time_point>
        last_request_times;
    auto now = std::chrono::steady_clock::now();

    bool rate_limited = false;
    {
        std::lock_guard<std::mutex> lock(request_times_mutex);
        
        // Check if this client is requesting too quickly
        auto it = last_request_times.find(c);
        if (it != last_request_times.end()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - it->second)
                               .count();
            if (elapsed < 25) {
                // Too many requests, will respond with a 429 (Too Many Requests)
                LOGW(
                    "Rate limit: connection requesting too quickly ({}ms), camera "
                    "{}",
                    elapsed, camera_id);
                rate_limited = true;
            }
        }
        
        if (!rate_limited) {
            // Update the last request time
            last_request_times[c] = now;
            
            // Clean up connection-specific data for connections that are too old (5 minutes)
            auto clean_time = now - std::chrono::minutes(5);
            for (auto it = last_request_times.begin(); it != last_request_times.end();) {
                if (it->second < clean_time) {
                    it = last_request_times.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    
    if (rate_limited) {
        mg_http_reply(c, 429, "Access-Control-Allow-Origin: *\r\n",
                      "Too many requests. Please wait at least 25ms "
                      "between requests.");
        return;
    }

    try {
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
        } else {
            // Not found or error encoding
            LOGE("No valid frame available for camera {}", camera_id);
            mg_http_reply(c, 404, "Access-Control-Allow-Origin: *\r\n",
                          "Camera not found or no frame available");
        }
    } catch (const std::exception& e) {
        LOGE("Error serving frame for camera {}: {}", camera_id, e.what());
        mg_http_reply(c, 500, "Access-Control-Allow-Origin: *\r\n",
                      "Error processing camera frame");
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
                // Process events with shorter timeout for lower latency
                mg_mgr_poll(&mgr_, 5);  // 5ms timeout for more responsive HTTP handling
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

void StreamService::setFrameProcessingRate(int every_n_frames) {
    if (every_n_frames < 1) {
        LOGW("Invalid frame processing rate {}, using 1", every_n_frames);
        process_every_n_frames_ = 1;
    } else {
        process_every_n_frames_ = every_n_frames;
        LOGI("Set to process every {} frames for better performance", process_every_n_frames_);
    }
}

std::expected<void, std::string> StreamService::tick() {
    // Lock for thread safety when updating latest frames
    std::lock_guard<std::mutex> lock(mutex_);

    // Flag to track if we got frames from any queue
    bool any_frames_received = false;

    // Process frames from all camera queues
    for (auto& [camera_id, queue] : camera_queues_) {
        cv::Mat frame;
        if (queue->try_pop_zero_copy(frame)) {
            // Process frame and store the latest frame for each camera
            LOGI("New frame received from camera {}", camera_id);

            // Store the latest frame - we need to make a deep copy to ensure it's not overwritten
            // For detection to work reliably, we need stable frames that don't change
            if (!frame.empty()) {
                latest_frames_[camera_id] = frame.clone(); // Make a deep copy for stability
                LOGD("Stored new frame for camera {} ({}x{})", 
                    camera_id, latest_frames_[camera_id].cols, latest_frames_[camera_id].rows);
            } else {
                LOGW("Received empty frame from camera {}, ignoring", camera_id);
            }
            any_frames_received = true;
            
            // Run person detection if enabled, but at a reduced rate
            // Only run detection if we have a valid frame and the detector is enabled
            if (use_person_detector_ && yolo_ && latest_frames_.find(camera_id) != latest_frames_.end() && 
                !latest_frames_[camera_id].empty()) {
                
                // Skip frames based on counter for better performance
                bool should_run_detection = (frame_counter_++ % process_every_n_frames_ == 0);
                
                // If active_detection_camera is set, only run detection on that camera
                if (!active_detection_camera_.empty() && camera_id != active_detection_camera_) {
                    should_run_detection = false;
                }
                
                // Add time-based throttling on top of frame skipping
                static std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_detection_time;
                static constexpr int DETECTION_INTERVAL_MS = 200; // Run detection at ~5fps
                
                auto now = std::chrono::steady_clock::now();
                
                // Check if we've run detection recently for this camera
                auto time_it = last_detection_time.find(camera_id);
                if (time_it != last_detection_time.end() && should_run_detection) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - time_it->second).count();
                    should_run_detection = elapsed >= DETECTION_INTERVAL_MS;
                }
                
                if (should_run_detection) {
                    try {
                        // Update detection time
                        last_detection_time[camera_id] = now;
                        
                        // YOLO detection needs properly formatted frames
                        // Always make a deep copy for detection to ensure proper memory layout
                        cv::Mat detection_frame;
                        int target_size = 416; // Optimal size for YOLO
                        double scale_factor = 1.0;
                        
                        // Use the stored frame copy that we confirmed is valid
                        cv::Mat& current_frame = latest_frames_[camera_id];
                        
                        // Double check the frame is still valid
                        if (current_frame.empty()) {
                            LOGE("Empty stored frame for camera {}, skipping YOLO detection", camera_id);
                            continue;
                        }
                        
                        LOGI("Processing detection for camera {} with frame size {}x{}", 
                             camera_id, current_frame.cols, current_frame.rows);
                        
                        // Ensure we have a proper continuous BGR image for YOLO
                        cv::Mat continuous_frame;
                        try {
                            if (!current_frame.isContinuous()) {
                                // Make a continuous copy for proper processing
                                current_frame.copyTo(continuous_frame);
                            } else {
                                // Frame is already continuous but we'll still make a copy
                                // to ensure memory layout compatibility with YOLO
                                continuous_frame = current_frame.clone();
                            }
                            
                            // Verify the copy worked
                            if (continuous_frame.empty()) {
                                LOGE("Failed to create continuous frame");
                                continue;
                            }
                        } catch (const cv::Exception& e) {
                            LOGE("OpenCV error creating continuous frame: {}", e.what());
                            continue;
                        }
                        
                        // Resize for optimal YOLO processing
                        if (continuous_frame.cols <= target_size && continuous_frame.rows <= target_size) {
                            // Small enough, but still need a properly formatted copy
                            detection_frame = continuous_frame.clone();
                        } else {
                            // Need to resize - compute scale factors
                            scale_factor = std::min(
                                static_cast<double>(target_size) / continuous_frame.cols,
                                static_cast<double>(target_size) / continuous_frame.rows);
                            int new_width = static_cast<int>(continuous_frame.cols * scale_factor);
                            int new_height = static_cast<int>(continuous_frame.rows * scale_factor);
                            
                            // Resize to target size
                            cv::resize(continuous_frame, detection_frame, cv::Size(new_width, new_height), 
                                      0, 0, cv::INTER_LINEAR);
                        }
                        
                        // Make sure we have a valid frame before trying conversions
                        if (detection_frame.empty()) {
                            LOGE("Empty detection frame, skipping YOLO detection");
                            continue;
                        }
                        
                        // Create a proper BGR copy for YOLO processing
                        // This ensures consistent memory layout and format
                        cv::Mat bgr_detection_frame;
                        
                        try {
                            // Check what format we're working with
                            if (detection_frame.channels() == 3) {
                                // For 3-channel images, clone to ensure proper format
                                // This avoids potential memory layout issues
                                bgr_detection_frame = detection_frame.clone();
                            } else if (detection_frame.channels() == 1) {
                                // Convert grayscale to BGR
                                cv::cvtColor(detection_frame, bgr_detection_frame, cv::COLOR_GRAY2BGR);
                            } else {
                                // Handle unexpected format by creating a blank BGR image
                                LOGW("Unexpected image format with {} channels", detection_frame.channels());
                                bgr_detection_frame = cv::Mat(detection_frame.rows, detection_frame.cols, CV_8UC3, cv::Scalar(0, 0, 0));
                            }
                        } catch (const cv::Exception& e) {
                            LOGE("OpenCV error during format conversion: {}", e.what());
                            continue;
                        }
                        
                        // Run YOLO detection with default thresholds
                        const float confidence_threshold = 0.25f;
                        const float iou_threshold = 0.45f;
                        
                        // Log detailed information about the frame being passed to YOLO
                        LOGI("Detecting on frame: type={}, size={}x{}, channels={}, continuous={}, empty={}",
                             bgr_detection_frame.type(), bgr_detection_frame.cols, bgr_detection_frame.rows,
                             bgr_detection_frame.channels(), 
                             bgr_detection_frame.isContinuous() ? "yes" : "no",
                             bgr_detection_frame.empty() ? "yes" : "no");
                            
                        // Additional logging for memory alignment and structure which can affect YOLO
                        LOGI("Frame memory: step={}, elemSize={}, total={}",
                             bgr_detection_frame.step[0], 
                             bgr_detection_frame.elemSize(),
                             bgr_detection_frame.total());
                        
                        std::vector<Detection> detections;
                        try {
                            // Attempt detection with better error handling
                            detections = yolo_->detect(bgr_detection_frame, confidence_threshold, iou_threshold);
                            LOGI("Detection successful - found {} objects", detections.size());
                        } catch (const std::exception& e) {
                            LOGE("YOLO detection failed with exception: {}", e.what());
                            continue;
                        }
                        
                        // If we resized, scale detections back to original size 
                        if (scale_factor != 1.0) {
                            // No need to recalculate scales - just use the inverse of scale_factor
                            double inverse_scale = 1.0 / scale_factor;
                            
                            for (auto& detection : detections) {
                                detection.box.center.x *= inverse_scale;
                                detection.box.center.y *= inverse_scale;
                                detection.box.width *= inverse_scale;
                                detection.box.height *= inverse_scale;
                            }
                        }
                        
                        // Store detections directly (no extra copy)
                        latest_detections_[camera_id] = std::move(detections);
                        
                        // Log only occasionally to reduce overhead
                        static int log_counter = 0;
                        if (++log_counter % 10 == 0) {
                            // Count person detections
                            int person_count = 0;
                            for (const auto& detection : latest_detections_[camera_id]) {
                                if (detection.class_id == 0) { // Person class
                                    person_count++;
                                }
                            }
                            
                            if (!latest_detections_[camera_id].empty()) {
                                LOGI("Detected {} objects ({} people) in camera {}",
                                     latest_detections_[camera_id].size(), person_count, camera_id);
                            }
                        }
                    } catch (const std::exception& e) {
                        // Log errors less frequently
                        static int error_counter = 0;
                        if (++error_counter % 10 == 0) {
                            LOGE("Error during person detection: {}", e.what());
                        }
                    }
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

                // Move the test frame directly to avoid any copying
                latest_frames_[camera_id] = std::move(test_frame);
                LOGI("Updated test frame for camera {}", camera_id);
            }
        }
    }

    // Reduce memory usage by cleaning up old cached frames
    {
        std::lock_guard<std::mutex> cache_lock(frame_cache_mutex);
        auto now = std::chrono::steady_clock::now();
        for (auto it = frame_cache.begin(); it != frame_cache.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second.timestamp).count();
            if (elapsed > 5000) { // Remove frames older than 5 seconds
                it = frame_cache.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Smaller sleep to avoid busy waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return {};
}

void StreamService::serveLatestFrame(const std::string& camera_id,
                                     std::vector<uint8_t>& jpeg_buffer) {
    // Note: This method should always be called with mutex_ locked by the caller
    
    // Ensure jpeg_buffer is empty at the start
    jpeg_buffer.clear();
    
    // Check frame cache first (under its own mutex to reduce contention)
    bool use_cached = false;
    {
        std::lock_guard<std::mutex> cache_lock(frame_cache_mutex);
        auto cache_it = frame_cache.find(camera_id);
        if (cache_it != frame_cache.end()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - cache_it->second.timestamp).count();
            if (elapsed < CACHE_TTL_MS) {
                // Use cached frame
                jpeg_buffer = cache_it->second.jpeg_data;
                use_cached = true;
            }
        }
    }
    
    if (use_cached) {
        // Used cached frame, exit early
        return;
    }
    
    auto it = latest_frames_.find(camera_id);
    if (it != latest_frames_.end() && !it->second.empty()) {
        try {
            // Validate the source frame first to ensure it's usable
            if (it->second.cols <= 0 || it->second.rows <= 0 || it->second.type() <= 0) {
                LOGE("Invalid frame dimensions or type for camera {}: {}x{} type={}", 
                     camera_id, it->second.cols, it->second.rows, it->second.type());
                throw std::runtime_error("Invalid frame dimensions or type");
            }
            
            // Create a clone of the frame to ensure it doesn't change during encoding
            // Must be a deep copy for thread safety
            cv::Mat original_frame;
            it->second.copyTo(original_frame); // Use copyTo for safer cloning
            
            // Use original frame directly if it's already the right size
            cv::Mat* frame_to_process = &original_frame;
            cv::Mat resized;
            
            // Original dimensions
            int orig_width = original_frame.cols;
            int orig_height = original_frame.rows;
            
            // Only resize if necessary
            if (orig_width > MAX_DISPLAY_WIDTH) {
                // Calculate aspect ratio and new size (maintain aspect ratio)
                int target_width = std::min(MAX_DISPLAY_WIDTH, orig_width);
                double aspect_ratio = static_cast<double>(orig_height) / orig_width;
                int target_height = static_cast<int>(target_width * aspect_ratio);
                
                // Fast resize using the most appropriate algorithm
                // Use INTER_NEAREST for higher speed if the size reduction is significant
                if (orig_width > target_width * 2 || orig_height > target_height * 2) {
                    cv::resize(original_frame, resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_NEAREST);
                } else {
                    // Use INTER_AREA for better quality when downsampling slightly
                    cv::resize(original_frame, resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);
                }
                frame_to_process = &resized;
            }
            
            // Flag to track if we have detections
            bool has_detections = false;
            
            // Draw bounding boxes for detections if enabled (directly on the frame we're processing)
            if (use_person_detector_ && yolo_) {
                auto detection_it = latest_detections_.find(camera_id);
                if (detection_it != latest_detections_.end() && !detection_it->second.empty()) {
                    try {
                        // Only scale bounding boxes if we resized the frame
                        // Create a drawing frame - note: we need to store this in a variable 
                        // that doesn't go out of scope before encoding
                        resized = frame_to_process->clone(); // Reuse resized variable to avoid dangling pointers
                        
                        if (frame_to_process == &original_frame) {
                            // Draw all detections on the frame directly
                            // No scaling needed since we're using original coordinates
                            yolo_->drawBoundingBox(resized, detection_it->second);
                        } else {
                            // Scale detections to match the current frame size
                            std::vector<Detection> scaled_detections;
                            double scale_x = static_cast<double>(resized.cols) / orig_width;
                            double scale_y = static_cast<double>(resized.rows) / orig_height;
                            
                            for (const auto& detection : detection_it->second) {
                                Detection scaled = detection;
                                scaled.box.center.x *= scale_x;
                                scaled.box.center.y *= scale_y;
                                scaled.box.width *= scale_x;
                                scaled.box.height *= scale_y;
                                scaled_detections.push_back(scaled);
                            }
                            
                            // Draw on the resized frame
                            yolo_->drawBoundingBox(resized, scaled_detections);
                        }
                        
                        // Use the drawing frame for further processing
                        frame_to_process = &resized;
                        
                        has_detections = true;
                        
                        // Log only occasionally to reduce overhead
                        static int log_counter = 0;
                        if (++log_counter % 30 == 0) {
                            LOGI("Drew {} detection boxes on frame for camera {}", 
                                detection_it->second.size(), camera_id);
                        }
                    } catch (const std::exception& e) {
                        // Log error but continue without boxes
                        LOGE("Error drawing detection boxes: {}", e.what());
                    }
                }
            }
            
            // Encode to JPEG with optimized settings for streaming
            std::vector<int> params = {
                cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY_STREAMING,
                cv::IMWRITE_JPEG_OPTIMIZE, 1,  // Enable optimization
                cv::IMWRITE_JPEG_PROGRESSIVE, 0  // Disable progressive (faster)
            };
            
            // Ensure we're using a valid Mat for encoding
            // If frame_to_process is a pointer to a local variable that will go out of scope,
            // we need to be careful
            if (frame_to_process && !frame_to_process->empty()) {
                cv::imencode(".jpg", *frame_to_process, jpeg_buffer, params);
            } else {
                // Fallback to original frame if frame_to_process is invalid
                LOGW("Invalid frame for encoding, using original frame");
                cv::imencode(".jpg", original_frame, jpeg_buffer, params);
            }
            
            // Only log once in a while to reduce overhead
            static int encode_log_counter = 0;
            if (++encode_log_counter % 100 == 0) {
                LOGI("Encoded frame {}x{} → {}x{}, size: {} bytes",
                    orig_width, orig_height, frame_to_process->cols, frame_to_process->rows, jpeg_buffer.size());
            }
            
            // Cache the encoded frame
            {
                std::lock_guard<std::mutex> cache_lock(frame_cache_mutex);
                frame_cache[camera_id] = {
                    jpeg_buffer,
                    std::chrono::steady_clock::now(),
                    orig_width,
                    orig_height,
                    has_detections
                };
            }
            
            return;
        } catch (const std::exception& e) {
            LOGE("Error processing frame: {}", e.what());
            jpeg_buffer.clear();
            // Fall through to fallback frame
        }
    } 
    
    // Use cached fallback frame if possible
    static std::mutex fallback_mutex;
    static std::unordered_map<std::string, std::vector<uint8_t>> fallback_frames;
    static std::chrono::steady_clock::time_point last_fallback_update;
    static constexpr int FALLBACK_UPDATE_MS = 1000; // Update fallback every second
    
    {
        std::lock_guard<std::mutex> lock(fallback_mutex);
        
        // Check if we should update the fallback frames
        auto now = std::chrono::steady_clock::now();
        bool update_fallback = fallback_frames.empty() || 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_fallback_update).count() > FALLBACK_UPDATE_MS;
        
        // If we have a cached fallback and don't need to update, use it
        if (!update_fallback && fallback_frames.find(camera_id) != fallback_frames.end()) {
            jpeg_buffer = fallback_frames[camera_id];
            return;
        }
        
        // Otherwise generate a new fallback frame
        try {
            // Simple test pattern (smaller and faster to generate)
            cv::Mat test_pattern(240, 320, CV_8UC3, cv::Scalar(100, 0, 200));
            
            // Simple rectangle
            cv::rectangle(test_pattern, cv::Point(50, 50), cv::Point(270, 190),
                          cv::Scalar(200, 200, 255), 2);
            
            // Minimal text
            cv::putText(test_pattern, "No Feed - " + camera_id, cv::Point(60, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
            
            // Get current time (only if updating)
            if (update_fallback) {
                std::time_t now_t = std::time(nullptr);
                char time_str[20];
                std::strftime(time_str, sizeof(time_str), "%H:%M:%S", std::localtime(&now_t));
                cv::putText(test_pattern, time_str, cv::Point(100, 150),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
                
                last_fallback_update = now;
            }
            
            // Encode with lower quality
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
            cv::imencode(".jpg", test_pattern, jpeg_buffer, params);
            
            // Cache the fallback
            fallback_frames[camera_id] = jpeg_buffer;
            
        } catch (const std::exception& e) {
            LOGE("Error creating fallback: {}", e.what());
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
