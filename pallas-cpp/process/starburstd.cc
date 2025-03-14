#include <service/camera_service.h>
#include <service/ps3_camera_service.h>
#include <chrono>
#include <memory>
#include <core/logger.h>

int webcam()
{
	const pallas::CameraServiceConfig config{
		.base = {.name = "starburst-webcam", .port = 8888, .interval_ms = 33.3}, // ~30fps
		.shared_memory_name = "camera-1",
		.shared_memory_frame_capacity = 20}; // Increased capacity for better buffering
	pallas::CameraService camera_service{config}; 

	camera_service.start();

	std::this_thread::sleep_for(std::chrono::seconds(300));
	camera_service.stop();
  
	return 0; 
}

int ps3()
{
	const pallas::PS3CameraServiceConfig config{
		.base = {.name = "starburst-ps3", .port = 8888, .interval_ms = 33.3}, // ~30fps
		.shared_memory_name = "camera-1",
		.shared_memory_frame_capacity = 60}; // Increased capacity for better buffering
	pallas::PS3CameraService camera_service{config}; 

	if (!camera_service.start()) {
		LOGE("Failed to start PS3 camera service");
		return 1;
	}
	
	LOGI("PS3 camera service started successfully, will run for 24 hours");
	// Run for 24 hours instead of 5 minutes
	std::this_thread::sleep_for(std::chrono::hours(24));
	camera_service.stop();
  
	return 0; 
}

int main() {
  pallas::init_logging();

  const bool is_ps3 = true;

  return is_ps3 ? ps3() : webcam(); 
}
