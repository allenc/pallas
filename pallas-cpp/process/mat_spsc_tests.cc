#include <core/mat_queue.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

using namespace pallas;
using namespace std::chrono_literals;

// Print first few bytes of matrix data for debugging
void print_matrix_data(const cv::Mat& mat, const std::string& name) {
    if (mat.empty()) {
        std::cout << name << " is empty" << std::endl;
        return;
    }
    
    std::cout << name << " (" << mat.cols << "x" << mat.rows << " type=" << mat.type() << "):" << std::endl;
    
    const uchar* data = mat.data;
    size_t total_bytes = mat.total() * mat.elemSize();
    size_t bytes_to_show = std::min(total_bytes, size_t(64));
    
    std::cout << "  Data address: " << static_cast<const void*>(data) << std::endl;
    std::cout << "  Total bytes: " << total_bytes << std::endl;
    std::cout << "  First " << bytes_to_show << " bytes: ";
    
    for (size_t i = 0; i < bytes_to_show; i++) {
        std::cout << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(data[i]) << " ";
        if ((i + 1) % 16 == 0 && i < bytes_to_show - 1) {
            std::cout << std::endl << "                   ";
        }
    }
    std::cout << std::dec << std::endl;
    
    // Save raw bytes to file for external comparison
    std::ofstream file(name + ".bin", std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(data), total_bytes);
        std::cout << "  Raw data saved to " << name << ".bin" << std::endl;
    }
}

// Generate test patterns for grayscale and color
cv::Mat create_grayscale_test_pattern(int rows, int cols) {
    cv::Mat mat(rows, cols, CV_8UC1, cv::Scalar(0));
    
    // Add a border
    for (int r = 0; r < rows; r++) {
        mat.at<uchar>(r, 0) = 255;
        mat.at<uchar>(r, cols-1) = 255;
    }
    for (int c = 0; c < cols; c++) {
        mat.at<uchar>(0, c) = 255;
        mat.at<uchar>(rows-1, c) = 255;
    }
    
    // Add a diagonal
    for (int i = 0; i < std::min(rows, cols); i++) {
        mat.at<uchar>(i, i) = 255;
    }
    
    // Add some patterns inside
    for (int r = 2; r < rows-2; r += 2) {
        for (int c = 2; c < cols-2; c += 2) {
            mat.at<uchar>(r, c) = 200;
        }
    }
    
    return mat;
}

cv::Mat create_color_test_pattern(int rows, int cols) {
    cv::Mat mat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Add a border (white)
    for (int r = 0; r < rows; r++) {
        mat.at<cv::Vec3b>(r, 0) = cv::Vec3b(255, 255, 255);
        mat.at<cv::Vec3b>(r, cols-1) = cv::Vec3b(255, 255, 255);
    }
    for (int c = 0; c < cols; c++) {
        mat.at<cv::Vec3b>(0, c) = cv::Vec3b(255, 255, 255);
        mat.at<cv::Vec3b>(rows-1, c) = cv::Vec3b(255, 255, 255);
    }
    
    // Add a diagonal (red)
    for (int i = 0; i < std::min(rows, cols); i++) {
        mat.at<cv::Vec3b>(i, i) = cv::Vec3b(0, 0, 255);  // Red in BGR
    }
    
    // Add some patterns inside (green)
    for (int r = 2; r < rows-2; r += 2) {
        for (int c = 2; c < cols-2; c += 2) {
            mat.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0);  // Green in BGR
        }
    }
    
    return mat;
}

// Perform byte-by-byte comparison
bool compare_matrices(const cv::Mat& original, const cv::Mat& received) {
    if (original.size() != received.size() || original.type() != received.type()) {
        std::cout << "Matrix size or type mismatch" << std::endl;
        return false;
    }
    
    int differing_bytes = 0;
    size_t total_bytes = original.total() * original.elemSize();
    
    for (size_t i = 0; i < total_bytes; i++) {
        if (original.data[i] != received.data[i]) {
            differing_bytes++;
        }
    }
    
    std::cout << "Comparison results:" << std::endl;
    std::cout << "  Total bytes: " << total_bytes << std::endl;
    std::cout << "  Differing bytes: " << differing_bytes << " ("
              << (100.0 * differing_bytes / total_bytes) << "%)" << std::endl;
    
    return differing_bytes == 0;
}

// Producer thread function for grayscale
void producer_grayscale_thread(const std::string& queue_name, int test_size) {
    try {
        // Create a simple grayscale test pattern
        cv::Mat test_mat = create_grayscale_test_pattern(test_size, test_size);
        
        // Display and save the original matrix
        print_matrix_data(test_mat, "original_grayscale_matrix");
        cv::imwrite("original_grayscale_matrix.png", test_mat);
        
        // Create the queue
        MatQueue::Close(queue_name); // Ensure clean state
        auto queue = MatQueue::Create(queue_name, 1);
        
        std::cout << "Grayscale Producer: Pushing test matrix (" 
                  << test_mat.cols << "x" << test_mat.rows 
                  << " type=" << test_mat.type() << ")" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool result = queue.try_push(test_mat);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (result) {
            std::cout << "Grayscale Producer: Push successful (" << duration << "ms)" << std::endl;
        } else {
            std::cout << "Grayscale Producer: Push failed" << std::endl;
        }
        
        std::cout << "Grayscale Producer: Done" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Grayscale Producer exception: " << e.what() << std::endl;
    }
}

// Producer thread function for color
void producer_color_thread(const std::string& queue_name, int test_size) {
    try {
        // Create a simple color test pattern
        cv::Mat test_mat = create_color_test_pattern(test_size, test_size);
        
        // Display and save the original matrix
        print_matrix_data(test_mat, "original_color_matrix");
        cv::imwrite("original_color_matrix.png", test_mat);
        
        // Create the queue
        MatQueue::Close(queue_name); // Ensure clean state
        auto queue = MatQueue::Create(queue_name, 1);
        
        std::cout << "Color Producer: Pushing test matrix (" 
                  << test_mat.cols << "x" << test_mat.rows 
                  << " type=" << test_mat.type() << ")" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool result = queue.try_push(test_mat);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (result) {
            std::cout << "Color Producer: Push successful (" << duration << "ms)" << std::endl;
        } else {
            std::cout << "Color Producer: Push failed" << std::endl;
        }
        
        std::cout << "Color Producer: Done" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Color Producer exception: " << e.what() << std::endl;
    }
}

// Consumer thread function for grayscale
void consumer_grayscale_thread(const std::string& queue_name) {
    try {
        // Wait for producer to initialize
        std::this_thread::sleep_for(500ms);
        
        auto queue = MatQueue::Open(queue_name);
        cv::Mat received;
        bool got_matrix = false;
        
        std::cout << "Grayscale Consumer: Waiting for matrix" << std::endl;
        
        // Try several times with timeout
        for (int attempts = 0; attempts < 10 && !got_matrix; attempts++) {
            got_matrix = queue.try_pop(received);
            
            if (got_matrix) {
                std::cout << "Grayscale Consumer: Pop successful" << std::endl;
                
                // Analyze the received data
                print_matrix_data(received, "received_grayscale_matrix");
                
                // Save image for visual inspection
                cv::imwrite("received_grayscale_matrix.png", received);
                
                // Read original image for comparison
                cv::Mat original = cv::imread("original_grayscale_matrix.png", cv::IMREAD_UNCHANGED);
                
                // Compare matrices
                if (!original.empty()) {
                    bool comparison_result = compare_matrices(original, received);
                    std::cout << "Grayscale Matrix Comparison: " 
                              << (comparison_result ? "PASS" : "FAIL") << std::endl;
                    
                    // Generate and save difference image if needed
                    if (!comparison_result) {
                        cv::Mat diff;
                        cv::absdiff(original, received, diff);
                        diff *= 5;  // Enhance visibility
                        cv::imwrite("grayscale_difference_matrix.png", diff);
                    }
                }
            } else {
                std::this_thread::sleep_for(100ms);
            }
        }
        
        if (!got_matrix) {
            std::cout << "Grayscale Consumer: Failed to receive matrix after multiple attempts" << std::endl;
        }
        
        std::cout << "Grayscale Consumer: Done" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Grayscale Consumer exception: " << e.what() << std::endl;
    }
}

// Consumer thread function for color
void consumer_color_thread(const std::string& queue_name) {
    try {
        // Wait for producer to initialize
        std::this_thread::sleep_for(500ms);
        
        auto queue = MatQueue::Open(queue_name);
        cv::Mat received;
        bool got_matrix = false;
        
        std::cout << "Color Consumer: Waiting for matrix" << std::endl;
        
        // Try several times with timeout
        for (int attempts = 0; attempts < 10 && !got_matrix; attempts++) {
            got_matrix = queue.try_pop(received);
            
            if (got_matrix) {
                std::cout << "Color Consumer: Pop successful" << std::endl;
                
                // Analyze the received data
                print_matrix_data(received, "received_color_matrix");
                
                // Save image for visual inspection
                cv::imwrite("received_color_matrix.png", received);
                
                // Read original image for comparison
                cv::Mat original = cv::imread("original_color_matrix.png", cv::IMREAD_UNCHANGED);
                
                // Compare matrices
                if (!original.empty()) {
                    bool comparison_result = compare_matrices(original, received);
                    std::cout << "Color Matrix Comparison: " 
                              << (comparison_result ? "PASS" : "FAIL") << std::endl;
                    
                    // Generate and save difference image if needed
                    if (!comparison_result) {
                        cv::Mat diff;
                        cv::absdiff(original, received, diff);
                        diff *= 5;  // Enhance visibility
                        cv::imwrite("color_difference_matrix.png", diff);
                    }
                }
            } else {
                std::this_thread::sleep_for(100ms);
            }
        }
        
        if (!got_matrix) {
            std::cout << "Color Consumer: Failed to receive matrix after multiple attempts" << std::endl;
        }
        
        std::cout << "Color Consumer: Done" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Color Consumer exception: " << e.what() << std::endl;
    }
}

int main() {
    const std::string GRAYSCALE_QUEUE_NAME = "test_grayscale_queue";
    const std::string COLOR_QUEUE_NAME = "test_color_queue";
    
    // Use a small test matrix for easier debugging (64x64 pixels)
    const int TEST_SIZE = 64;
    
    // Test grayscale
    {
        std::thread producer(producer_grayscale_thread, GRAYSCALE_QUEUE_NAME, TEST_SIZE);
        std::thread consumer(consumer_grayscale_thread, GRAYSCALE_QUEUE_NAME);
        
        // Wait for threads to finish
        producer.join();
        consumer.join();
        
        // Cleanup
        MatQueue::Close(GRAYSCALE_QUEUE_NAME);
    }
    
    // Test color
    {
        std::thread producer(producer_color_thread, COLOR_QUEUE_NAME, TEST_SIZE);
        std::thread consumer(consumer_color_thread, COLOR_QUEUE_NAME);
        
        // Wait for threads to finish
        producer.join();
        consumer.join();
        
        // Cleanup
        MatQueue::Close(COLOR_QUEUE_NAME);
    }
    
    return 0;
}
