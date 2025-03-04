#include <base64.h>  // Include a base64 encoding library like https://github.com/ReneNyffenegger/cpp-base64
#include <curl/curl.h>

#include <expected>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

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
std::expected<void, std::string> text_user(const PhoneNumber& number,
                                           const Payload& payload) {
    // Twilio credentials - in a real application, these should be securely
    // stored/accessed
    const std::string TWILIO_ACCOUNT_SID = "YOUR_ACCOUNT_SID";
    const std::string TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN";
    const std::string TWILIO_PHONE_NUMBER =
        "YOUR_TWILIO_PHONE_NUMBER";  // Must be in E.164 format: +1XXXYYYZZZZ

    // Initialize curl
    CURL* curl = curl_easy_init();
    if (!curl) {
        return std::unexpected<std::string>("Failed to initialize CURL");
    }

    // Prepare the request URL
    std::string url = "https://api.twilio.com/2010-04-01/Accounts/" +
                      TWILIO_ACCOUNT_SID + "/Messages.json";

    // Prepare the data
    std::string postFields =
        "From=" + TWILIO_PHONE_NUMBER + "&To=" + number.toString() +
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
                     (TWILIO_ACCOUNT_SID + ":" + TWILIO_AUTH_TOKEN).c_str());

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

// Example usage
int main() {
    // Create a phone number
    PhoneNumber recipient("+15551234567");

    // Create a payload with a message and two images
    Payload payload;
    payload.message = "Hello from C++!";

    // Create sample images (in a real app, you'd load these from files)
    cv::Mat image1(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
    cv::Mat image2(100, 100, CV_8UC3, cv::Scalar(0, 255, 0));
    payload.images.push_back(image1);
    payload.images.push_back(image2);

    // Send the message
    auto result = text_user(recipient, payload);

    if (result.has_value()) {
        std::cout << "Message sent successfully!" << std::endl;
    } else {
        std::cout << "Failed to send message: " << result.error() << std::endl;
    }

    return 0;
}
