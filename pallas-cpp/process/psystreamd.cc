#include <service/inference_service.h>
#include <chrono>
#include <memory>
#include <filesystem>

#include <spdlog/spdlog.h>

int main() {
  spdlog::set_level(spdlog::level::debug);

  const std::filesystem::path assets_path = "../assets/";
  
  const pallas::InferenceServiceConfig config{
      .base = {.name = "psystream", .port = 8888, .interval_ms = 50.0},
	  .inference {
		  .use_gpu = false,
		  .yolo_path = assets_path / "yolo11.onnx",
		  .yolo_labels_path = assets_path / "yolo11_labels.txt",
		  .sam_encoder_path = assets_path  / "sam2.1_tiny_preprocess.onnx",
		  .sam_decoder_path = assets_path / "sam2.1_tiny.onnx",	  
		  .shared_memory_names = {"camera-1"}}};
  pallas::InferenceService inference_service{config};

  inference_service.start();

  std::this_thread::sleep_for(std::chrono::seconds(30)); 
  inference_service.stop();

  return 0;
}
