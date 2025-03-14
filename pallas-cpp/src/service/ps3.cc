#include "ps3.h"

#include <core/logger.h>
#include <opencv2/videoio.hpp>
#include <libusb-1.0/libusb.h>
#include <iostream>
#include <algorithm>
#include <array>

namespace pallas {

// PS3 Eye USB vendor ID and product ID
constexpr uint16_t PS3_EYE_VENDOR_ID = 0x1415;  // Sony
constexpr uint16_t PS3_EYE_PRODUCT_ID = 0x2000; // PS3 Eye

// Implementation class to handle the camera details
class PS3EyeCamera::Impl {
public:
    explicit Impl(const PS3EyeConfig& config) : config_(config), is_open_(false) {}

    ~Impl() {
        close();
    }

    std::expected<void, std::string> open() {
        try {
            // Try to open the camera with OpenCV (simplest approach for Linux)
            capture_.open(config_.device_id, cv::CAP_V4L2);
            
            if (!capture_.isOpened()) {
                return std::unexpected("Failed to open PS3 Eye camera with device ID " + 
                                      std::to_string(config_.device_id));
            }

            // Set camera properties
            capture_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
            capture_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
            capture_.set(cv::CAP_PROP_FPS, config_.fps);
            
            // Try to set auto gain
            capture_.set(cv::CAP_PROP_XI_GAIN, config_.auto_gain ? 1 : 0);
            
            // Set manual gain if auto gain is disabled
            if (!config_.auto_gain) {
                capture_.set(cv::CAP_PROP_GAIN, config_.gain);
            }
            
            // Set auto white balance
            capture_.set(cv::CAP_PROP_AUTO_WB, config_.auto_white_balance ? 1 : 0);
            
            is_open_ = true;
            LOGI("PS3 Eye camera opened successfully, deviceId={}, resolution={}x{}, fps={}",
                 config_.device_id, config_.width, config_.height, config_.fps);
                 
            return {};
        }
        catch (const std::exception& e) {
            return std::unexpected("Exception while opening PS3 Eye camera: " + 
                                  std::string(e.what()));
        }
    }

    bool isOpen() const {
        return is_open_ && capture_.isOpened();
    }

    void close() {
        if (is_open_) {
            capture_.release();
            is_open_ = false;
            LOGI("PS3 Eye camera closed, deviceId={}", config_.device_id);
        }
    }

    std::expected<cv::Mat, std::string> captureFrame() {
        if (!isOpen()) {
            return std::unexpected("PS3 Eye camera is not open");
        }

        cv::Mat frame;
        if (!capture_.read(frame)) {
            return std::unexpected("Failed to capture frame from PS3 Eye camera");
        }

        if (frame.empty()) {
            return std::unexpected("Captured empty frame from PS3 Eye camera");
        }

        // Apply horizontal/vertical flipping if needed
        if (config_.flip_horizontal || config_.flip_vertical) {
            int flip_code = 0;
            if (config_.flip_horizontal && !config_.flip_vertical) flip_code = 1;
            else if (!config_.flip_horizontal && config_.flip_vertical) flip_code = 0;
            else flip_code = -1; // both flips
            
            cv::flip(frame, frame, flip_code);
        }

        return frame;
    }

    bool setAutoGain(bool enable) {
        if (!isOpen()) return false;
        config_.auto_gain = enable;
        return capture_.set(cv::CAP_PROP_XI_GAIN, enable ? 1 : 0);
    }

    bool setGain(int gain) {
        if (!isOpen()) return false;
        config_.gain = gain;
        return capture_.set(cv::CAP_PROP_GAIN, gain);
    }

    bool setAutoWhiteBalance(bool enable) {
        if (!isOpen()) return false;
        config_.auto_white_balance = enable;
        return capture_.set(cv::CAP_PROP_AUTO_WB, enable ? 1 : 0);
    }

    bool setExposure(int exposure) {
        if (!isOpen()) return false;
        return capture_.set(cv::CAP_PROP_EXPOSURE, exposure);
    }

    bool setRedBalance(int red_balance) {
        if (!isOpen()) return false;
        return capture_.set(cv::CAP_PROP_WB_TEMPERATURE, red_balance);
    }

    bool setBlueBalance(int blue_balance) {
        if (!isOpen()) return false;
        // OpenCV doesn't have a direct property for blue balance
        // This is an approximation using color temperature
        return capture_.set(cv::CAP_PROP_WB_TEMPERATURE, blue_balance);
    }

    bool setFlip(bool horizontal, bool vertical) {
        config_.flip_horizontal = horizontal;
        config_.flip_vertical = vertical;
        return true;
    }

    const PS3EyeConfig& getConfig() const {
        return config_;
    }

private:
    PS3EyeConfig config_;
    cv::VideoCapture capture_;
    bool is_open_;
};

// Static method to get available PS3 Eye devices
std::vector<int> PS3EyeCamera::getDeviceList() {
    std::vector<int> devices;
    
    // Initialize libusb
    libusb_context* ctx = nullptr;
    if (libusb_init(&ctx) != 0) {
        LOGE("Failed to initialize libusb");
        return devices;
    }
    
    // Get device list
    libusb_device** device_list = nullptr;
    ssize_t count = libusb_get_device_list(ctx, &device_list);
    
    if (count < 0) {
        LOGE("Failed to get USB device list");
        libusb_exit(ctx);
        return devices;
    }
    
    // Find all PS3 Eye cameras
    std::array<int, 10> potential_devices{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // Try up to 10 video devices
    
    for (int device_id : potential_devices) {
        cv::VideoCapture temp(device_id, cv::CAP_V4L2);
        if (temp.isOpened()) {
            // Try to check if this is a PS3 Eye
            // There's no direct way to check in OpenCV, so we'll use rough heuristics
            
            // PS3 Eye should support 640x480 at 60fps
            temp.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            temp.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            temp.set(cv::CAP_PROP_FPS, 60);
            
            double actual_width = temp.get(cv::CAP_PROP_FRAME_WIDTH);
            double actual_height = temp.get(cv::CAP_PROP_FRAME_HEIGHT);
            double actual_fps = temp.get(cv::CAP_PROP_FPS);
            
            // Check if camera supports PS3 Eye typical resolution and framerate
            if (actual_width == 640 && actual_height == 480 && actual_fps >= 30) {
                devices.push_back(device_id);
                LOGI("Found potential PS3 Eye camera at device ID {}", device_id);
            }
            
            temp.release();
        }
    }
    
    // Look for PS3 Eye devices via USB as a backup method
    for (ssize_t i = 0; i < count; ++i) {
        libusb_device* device = device_list[i];
        libusb_device_descriptor desc;
        
        if (libusb_get_device_descriptor(device, &desc) == 0) {
            if (desc.idVendor == PS3_EYE_VENDOR_ID && 
                desc.idProduct == PS3_EYE_PRODUCT_ID) {
                
                uint8_t bus = libusb_get_bus_number(device);
                uint8_t address = libusb_get_device_address(device);
                
                LOGI("Found PS3 Eye camera on USB bus {} address {}", bus, address);
                
                // We found a PS3 Eye device, but OpenCV needs the /dev/video* ID
                // This is more of a notification than actual ID mapping
            }
        }
    }
    
    // Free device list and exit libusb context
    libusb_free_device_list(device_list, 1);
    libusb_exit(ctx);
    
    return devices;
}

// Implementation of the public interface methods

PS3EyeCamera::PS3EyeCamera(PS3EyeConfig config)
    : impl_(std::make_unique<Impl>(config)), config_(std::move(config)) {}

PS3EyeCamera::~PS3EyeCamera() = default;

std::expected<void, std::string> PS3EyeCamera::open() {
    return impl_->open();
}

bool PS3EyeCamera::isOpen() const {
    return impl_->isOpen();
}

void PS3EyeCamera::close() {
    impl_->close();
}

std::expected<cv::Mat, std::string> PS3EyeCamera::captureFrame() {
    return impl_->captureFrame();
}

bool PS3EyeCamera::setAutoGain(bool enable) {
    return impl_->setAutoGain(enable);
}

bool PS3EyeCamera::setGain(int gain) {
    return impl_->setGain(gain);
}

bool PS3EyeCamera::setAutoWhiteBalance(bool enable) {
    return impl_->setAutoWhiteBalance(enable);
}

bool PS3EyeCamera::setExposure(int exposure) {
    return impl_->setExposure(exposure);
}

bool PS3EyeCamera::setRedBalance(int red_balance) {
    return impl_->setRedBalance(red_balance);
}

bool PS3EyeCamera::setBlueBalance(int blue_balance) {
    return impl_->setBlueBalance(blue_balance);
}

bool PS3EyeCamera::setFlip(bool horizontal, bool vertical) {
    return impl_->setFlip(horizontal, vertical);
}

const PS3EyeConfig& PS3EyeCamera::getConfig() const {
    return config_;
}

} // namespace pallas
