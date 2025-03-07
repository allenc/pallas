#include <service/camera_service.h>
#include <chrono>
#include <memory>
#include <core/logger.h>

int main() {
  pallas::init_logging();

  const pallas::CameraServiceConfig config{
      .base = {.name = "starburst", .port = 8888, .interval_ms = 50.0},
      .shared_memory_name = "camera-1",
      .shared_memory_frame_capacity = 5};
  pallas::CameraService camera_service{config}; 

  camera_service.start();

  std::this_thread::sleep_for(std::chrono::seconds(300));
  camera_service.stop();

  return 0;
}
