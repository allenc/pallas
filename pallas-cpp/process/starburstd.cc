#include <service/camera_service.h>
#include <service/ps3_camera_service.h>
#include <chrono>
#include <memory>
#include <core/logger.h>
#include <string>
#include <iostream>

int webcam(int device_id = 0)
{
	// Create a unique shared memory name for this webcam instance
	std::string shared_memory_name = "webcam-" + std::to_string(device_id);
	
	const pallas::CameraServiceConfig config{
		.base = {.name = "starburst-webcam-" + std::to_string(device_id), .port = 8888 + device_id, .interval_ms = 16.6}, // ~60fps
		.shared_memory_name = shared_memory_name,
		.shared_memory_frame_capacity = 60}; // Higher capacity for 60fps streaming
	pallas::CameraService camera_service{config}; 

	camera_service.start();

	LOGI("Webcam service started successfully with device_id {} and shared memory {}, will run for 24 hours", device_id, shared_memory_name);
	std::this_thread::sleep_for(std::chrono::hours(24));
	camera_service.stop();
  
	return 0; 
}

int ps3(int device_id = 0)
{
	// Create a unique shared memory name for this PS3 camera instance
	std::string shared_memory_name = "ps3-" + std::to_string(device_id);
	
	const pallas::PS3CameraServiceConfig config{
		.base = {.name = "starburst-ps3-" + std::to_string(device_id), .port = 8888 + device_id, .interval_ms = 16.6}, // ~60fps
		.shared_memory_name = shared_memory_name,
		.shared_memory_frame_capacity = 120, // Higher capacity for 60fps streaming
		.camera_config = {.device_id = device_id} // Set the device_id in the camera_config
	}; 
	pallas::PS3CameraService camera_service{config}; 

	if (!camera_service.start()) {
		LOGE("Failed to start PS3 camera service with device_id {} and shared memory {}", device_id, shared_memory_name);
		return 1;
	}
	
	LOGI("PS3 camera service started successfully with device_id {} and shared memory {}, will run for 24 hours", device_id, shared_memory_name);
	std::this_thread::sleep_for(std::chrono::hours(24));
	camera_service.stop();
  
	return 0; 
}

void print_usage() {
	std::cout << "Usage: ./starburstd [options]\n"
			  << "Options:\n"
			  << "  --webcam <id>    Use webcam with specified device ID (default: 0)\n"
			  << "  --ps3 <id>       Use PS3 camera with specified device ID (default: 0)\n"
			  << "  --help           Display this help message\n"
			  << std::endl;
}

int main(int argc, char** argv) {
	pallas::init_logging();

	bool use_ps3 = false;
	int ps3_device_id = 0;
	bool use_webcam = false;
	int webcam_device_id = 0;

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--help") {
			print_usage();
			return 0;
		} else if (arg == "--webcam") {
			use_webcam = true;
			if (i + 1 < argc) {
				try {
					webcam_device_id = std::stoi(argv[++i]);
				} catch (const std::exception& e) {
					LOGE("Invalid webcam device ID: {}", argv[i]);
					print_usage();
					return 1;
				}
			}
		} else if (arg == "--ps3") {
			use_ps3 = true;
			if (i + 1 < argc) {
				try {
					ps3_device_id = std::stoi(argv[++i]);
				} catch (const std::exception& e) {
					LOGE("Invalid PS3 device ID: {}", argv[i]);
					print_usage();
					return 1;
				}
			}
		} else {
			LOGE("Unknown argument: {}", arg);
			print_usage();
			return 1;
		}
	}

	// If no camera type is specified, use PS3 as default
	if (!use_webcam && !use_ps3) {
		LOGI("No camera type specified, using PS3 camera by default");
		use_ps3 = true;
	}
	
	// If both camera types are specified, use the last one specified
	if (use_webcam && use_ps3) {
		LOGW("Both webcam and PS3 camera specified, using PS3 camera");
		use_webcam = false;
	}

	if (use_ps3) {
		LOGI("Starting PS3 camera with device_id: {}", ps3_device_id);
		return ps3(ps3_device_id);
	} else {
		LOGI("Starting webcam with device_id: {}", webcam_device_id);
		return webcam(webcam_device_id);
	}
}
