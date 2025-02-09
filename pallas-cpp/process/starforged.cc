#include <service/viewer_service.h>
#include <chrono>
#include <memory>
#include <core/logger.h>

int main() {
	pallas::init_logging();

  const pallas::ViewerServiceConfig config{
      .base = {.name = "starforged", .port = 8888, .interval_ms = 50.0},
      .shared_memory_names = {"camera-1"}};

  pallas::ViewerService viewer_service(config);
  viewer_service.start(); 

  std::this_thread::sleep_for(std::chrono::seconds(30)); 
  viewer_service.stop();

  return 0;
}
