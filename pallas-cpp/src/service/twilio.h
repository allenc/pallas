#pragma once

#include <curl/curl.h>

#include <expected>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

// Simple base64 encoding function
// You may replace this with a more robust implementation from a library
inline std::string base64_encode(const unsigned char* data, size_t length) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    
    std::string encoded;
    encoded.reserve(((length + 2) / 3) * 4);
    
    for (size_t i = 0; i < length; i += 3) {
        unsigned char b1 = data[i];
        unsigned char b2 = (i + 1 < length) ? data[i + 1] : 0;
        unsigned char b3 = (i + 2 < length) ? data[i + 2] : 0;
        
        encoded.push_back(base64_chars[(b1 >> 2) & 0x3F]);
        encoded.push_back(base64_chars[((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0F)]);
        encoded.push_back((i + 1 < length) ? base64_chars[((b2 & 0x0F) << 2) | ((b3 >> 6) & 0x03)] : '=');
        encoded.push_back((i + 2 < length) ? base64_chars[b3 & 0x3F] : '=');
    }
    
    return encoded;
}

// Define the PhoneNumber class
class PhoneNumber {
   private:
    std::string number;

   public:
    PhoneNumber(const std::string& num) : number(num) {
        // Optional: Add validation logic here
    }

    std::string toString() const { return number; }
};

// Define the Payload struct
struct Payload {
    std::string message;
    std::vector<cv::Mat> images;
};

// Callback function for CURL to write response
size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                     std::string* response) {
    size_t totalSize = size * nmemb;
    response->append((char*)contents, totalSize);
    return totalSize;
}

// Convert OpenCV Mat to base64 encoded string
std::string matToBase64(const cv::Mat& image) {
    std::vector<uchar> buffer;
    cv::imencode(".jpg", image, buffer);

    // Convert to base64
    std::string base64Image = base64_encode(buffer.data(), buffer.size());
    return base64Image;
}

// Main function to send text messages with optional images
std::expected<void, std::string> text_user(
    const PhoneNumber& number,
    const Payload& payload,
    const std::string& twilio_account_sid,
    const std::string& twilio_auth_token,
    const std::string& twilio_phone_number) {
    
    // Validate credentials
    if (twilio_account_sid.empty() || twilio_auth_token.empty() || twilio_phone_number.empty()) {
        return std::unexpected<std::string>("Missing Twilio credentials");
    }

    // Initialize curl
    CURL* curl = curl_easy_init();
    if (!curl) {
        return std::unexpected<std::string>("Failed to initialize CURL");
    }

    // Prepare the request URL
    std::string url = "https://api.twilio.com/2010-04-01/Accounts/" +
                      twilio_account_sid + "/Messages.json";

    // Prepare the data
    std::string postFields =
        "From=" + twilio_phone_number + "&To=" + number.toString() +
        "&Body=" + curl_easy_escape(curl, payload.message.c_str(), 0);

    // Add images if present
    if (!payload.images.empty()) {
        for (size_t i = 0; i < payload.images.size(); ++i) {
            // Convert image to base64
            std::string base64Image = matToBase64(payload.images[i]);

            // For MMS, we need to provide a URL that Twilio can access
            // In a real application, you'd upload the image somewhere (e.g.,
            // AWS S3) and provide the URL here. For simplicity, I'm showing how
            // to use Twilio's MediaUrl parameter
            std::string mediaParam =
                "&MediaUrl=" + curl_easy_escape(curl,
                                                ("https://example.com/image" +
                                                 std::to_string(i) + ".jpg")
                                                    .c_str(),
                                                0);

            postFields += mediaParam;

            // Note: In a real implementation, you would:
            // 1. Upload the image to a storage service (AWS S3, Google Cloud
            // Storage, etc.)
            // 2. Get the public URL for the uploaded image
            // 3. Pass that URL to Twilio's MediaUrl parameter
        }
    }

    // Set up the request
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postFields.c_str());
    curl_easy_setopt(curl, CURLOPT_USERPWD,
                     (twilio_account_sid + ":" + twilio_auth_token).c_str());

    // Set up response handling
    std::string response;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    curl_easy_cleanup(curl);

    // Check for errors
    if (res != CURLE_OK) {
        return std::unexpected<std::string>(
            "CURL error: " + std::string(curl_easy_strerror(res)));
    }

    if (httpCode < 200 || httpCode >= 300) {
        return std::unexpected<std::string>(
            "HTTP error: " + std::to_string(httpCode) +
            " Response: " + response);
    }

    return {};
}
