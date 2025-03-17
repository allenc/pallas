#include <core/logger.h>
#include <service/stream_service.h>

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace pallas;

static std::unique_ptr<StreamService> service;

void signal_handler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    if (service) service->stop();
    exit(signum);
}

// Create a simple standalone HTTP server using socket API
// This is a basic example for testing - it's not production-ready
// For a real product, use a proper HTTP server library
// Note: We now use the built-in HTTP server in StreamService
// This function is kept for backward compatibility with old code
void run_http_server(uint16_t port, StreamService& stream_service) {
    LOGI("This function is deprecated - using built-in HTTP server instead");
    
    // Just wait until the program is terminated
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }
}

int main(int argc, char* argv[]) {
    // Register signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize logging
    init_logging();
    LOGI("Starting streamd");
    
    LOGI("Usage: streamd [options]");
    LOGI("Options:");
    LOGI("  --port <port>                 : HTTP server port (default: 8080)");
    LOGI("  --shared-mem <name>              : Shared memory name (legacy, default: camera-1)");
    LOGI("  --camera-id <id>              : Camera ID (can be specified multiple times)");
    LOGI("  --use-person-detector         : Enable person detection with YOLO");
    LOGI("  --use-gpu                     : Use GPU for YOLO detection (requires CUDA)");
    LOGI("  --active-detection-only <id>  : Only run detection on selected camera ID");
    LOGI("  --yolo-model <path>           : Path to YOLO model (default: ../assets/yolo11.onnx)");
    LOGI("  --yolo-labels <path>          : Path to YOLO labels (default: ../assets/yolo11_labels.txt)");
    LOGI("");
    LOGI("Default camera IDs (automatically detected):");
    LOGI("  ps3-0, ps3-1      : PS3 cameras with device IDs 0 and 1");
    LOGI("  webcam-0, webcam-1 : Webcams with device IDs 0 and 1");

    // Parse command line arguments
    uint16_t port = 8080;
    std::string shared_mem_name = "camera-1";  // Default, kept for backward compatibility
    std::vector<std::string> camera_ids;
    bool use_person_detector = false;
    bool use_gpu = false;  // Default to CPU for compatibility
    std::string active_detection_camera = "";  // Empty means detect on all cameras
    std::string yolo_model_path = "../assets/yolo11.onnx";
    std::string yolo_labels_path = "../assets/yolo11_labels.txt";
    
    // Add all camera types by default (will connect to what's available)
    // These are the dynamically generated shared memory names from starburstd
    camera_ids.push_back("ps3-0");  // Default PS3 camera
    camera_ids.push_back("ps3-1");  // Second PS3 camera if available
    camera_ids.push_back("webcam-0"); // Default webcam
    camera_ids.push_back("webcam-1"); // Second webcam if available

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::atoi(argv[++i]));
        }
        else if (arg == "--shared-mem" && i + 1 < argc) {
            shared_mem_name = argv[++i];
        }
        else if (arg == "--camera-id" && i + 1 < argc) {
            if (camera_ids.size() > 0) {
                // Clear default camera IDs if user specifies their own
                camera_ids.clear();
            }
            camera_ids.push_back(argv[++i]);
        }
        else if (arg == "--use-person-detector") {
            use_person_detector = true;
        }
        else if (arg == "--use-gpu") {
            use_gpu = true;
            LOGI("GPU acceleration enabled for neural networks");
        }
        else if (arg == "--active-detection-only" && i + 1 < argc) {
            active_detection_camera = argv[++i];
            LOGI("Active detection mode enabled - will only run detection on camera: {}", active_detection_camera);
        }
        else if (arg == "--yolo-model" && i + 1 < argc) {
            yolo_model_path = argv[++i];
        }
        else if (arg == "--yolo-labels" && i + 1 < argc) {
            yolo_labels_path = argv[++i]; 
       }
    }
    
    LOGI("Command line parsing complete:");
    LOGI("  Port: {}", port);
    LOGI("  Shared memory: {}", shared_mem_name);
    LOGI("  Camera IDs: {}", fmt::join(camera_ids, ", "));
    LOGI("  Use person detector: {}", use_person_detector ? "Yes" : "No");
    if (use_person_detector) {
        LOGI("  YOLO model: {}", yolo_model_path);
        LOGI("  YOLO labels: {}", yolo_labels_path);
        LOGI("  GPU acceleration: {}", use_gpu ? "Enabled" : "Disabled");
        LOGI("  Active detection mode: {}", 
             !active_detection_camera.empty() ? active_detection_camera : "All cameras");
    }

    // Configure and start the service
    StreamServiceConfig config;
    config.base.name = "streamd";
    config.base.interval_ms = 16.6;  // ~60 FPS
    config.http_port = port;
    config.shared_memory_name = shared_mem_name;
    config.camera_ids = camera_ids;
    config.use_person_detector = use_person_detector;
    config.yolo_model_path = yolo_model_path;
    config.yolo_labels_path = yolo_labels_path;
    config.use_gpu = use_gpu; 

    LOGI("Configuration: port={}, shared_memory_name={}, camera_ids={}", 
         port, shared_mem_name, fmt::join(camera_ids, ","));
         
    // Test that starburstd is actually running
    LOGI("Testing for presence of starburstd process");
    FILE* fp = popen("pgrep starburstd", "r");
    if (fp) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), fp) != nullptr) {
            LOGI("Found starburstd process: {}", buffer);
        } else {
            LOGI("starburstd process not found, will generate test frames");
        }
        pclose(fp);
    }

    service = std::make_unique<StreamService>(config);
    
    // Set frame processing rate for better performance
    // Process every 3rd frame (adjust as needed)
    service->setFrameProcessingRate(3);
    
    if (!service->start()) {
        LOGE("Failed to start stream service");
        return 1;
    }

    LOGI("Stream service started successfully");
    
    // We now use a pre-built frontend file from the frontend directory
    LOGI("Using frontend file from: {}", std::filesystem::current_path().string() + "/frontend/index.html");
    
    // No need to start another HTTP server - StreamService already has one running
    LOGI("Using built-in HTTP server in StreamService");
    
    // Wait for a signal to terminate (Ctrl+C)
    std::cout << "Press Ctrl+C to exit" << std::endl;
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    LOGI("Exiting streamd");
    return 0;
}
