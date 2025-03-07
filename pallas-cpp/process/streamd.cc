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

    // Parse command line arguments
    // In a real application, you would add proper CLI argument parsing
    uint16_t port = 8080;
    std::string shared_mem_name = "camera-1";  // Changed to match starburstd
    std::vector<std::string> camera_ids = {"camera-1"};

    if (argc > 1) {
        port = static_cast<uint16_t>(std::atoi(argv[1]));
    }
    
    if (argc > 2) {
        shared_mem_name = argv[2];
    }
    
    if (argc > 3) {
        camera_ids.clear();
        for (int i = 3; i < argc; i++) {
            camera_ids.push_back(argv[i]);
        }
    }

    // Configure and start the service
    StreamServiceConfig config;
    config.base.name = "streamd";
    config.base.interval_ms = 33.0;  // ~30 FPS
    config.http_port = port;
    config.shared_memory_name = shared_mem_name;
    config.camera_ids = camera_ids;
    
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
