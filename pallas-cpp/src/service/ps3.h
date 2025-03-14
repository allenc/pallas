#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <expected>

namespace pallas {

struct PS3EyeConfig {
    int device_id{0};            // USB device ID
    int width{640};              // Frame width (default 640)
    int height{480};             // Frame height (default 480)
    int fps{60};                 // Frames per second (30 or 60 for PS3 Eye)
    bool auto_gain{true};        // Auto gain control
    int gain{20};                // Manual gain (if auto_gain is false)
    bool auto_white_balance{true}; // Auto white balance
    bool flip_horizontal{false}; // Flip image horizontally
    bool flip_vertical{false};   // Flip image vertically
};

class PS3EyeCamera {
public:
    // Get list of available PS3 Eye camera device IDs
    static std::vector<int> getDeviceList();
    
    // Create a PS3 Eye camera instance
    explicit PS3EyeCamera(PS3EyeConfig config = {});
    ~PS3EyeCamera();

    // Non-copyable
    PS3EyeCamera(const PS3EyeCamera&) = delete;
    PS3EyeCamera& operator=(const PS3EyeCamera&) = delete;

    // Open the camera
    std::expected<void, std::string> open();
    
    // Check if camera is open
    bool isOpen() const;
    
    // Close the camera
    void close();
    
    // Capture a frame (returns empty cv::Mat on failure)
    std::expected<cv::Mat, std::string> captureFrame();
    
    // Set camera properties
    bool setAutoGain(bool enable);
    bool setGain(int gain);
    bool setAutoWhiteBalance(bool enable);
    bool setExposure(int exposure);
    bool setRedBalance(int red_balance);
    bool setBlueBalance(int blue_balance);
    bool setFlip(bool horizontal, bool vertical);
    
    // Get current configuration
    const PS3EyeConfig& getConfig() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    PS3EyeConfig config_;
};

} // namespace pallas