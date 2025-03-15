#pragma once
#include <core/logger.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>

namespace pallas {

template <size_t MAX_FRAME_SIZE>
class MatQueue {
   private:
    struct alignas(64) MatHeader {
        int rows;
        int cols;
        int type;
        size_t data_size;
        size_t step;           // Step size for direct memory access
    };

    struct alignas(64) QueueHeader {
        std::atomic<size_t> write_pos;
        std::atomic<size_t> read_pos;
        std::atomic<bool> was_overwritten;
        size_t capacity;
    };

    void* mapped_memory_ = nullptr;
    QueueHeader* header_ = nullptr;
    uint8_t* buffer_ = nullptr;
    int fd_ = -1;
    std::string name_;
    size_t total_size_ = 0;

    bool copy_to_queue(const cv::Mat& mat, size_t position) {
        // Get a continuous version of the matrix if needed
        // This avoids copying if the matrix is already continuous
        const cv::Mat& continuous = mat.isContinuous() ? mat : mat.clone();
        size_t data_size = continuous.total() * continuous.elemSize();

        // Initialize header with step information for zero-copy access
        MatHeader mat_header{
            continuous.rows, 
            continuous.cols,
            continuous.type(), 
            data_size,
            continuous.step[0]    // Add step size for proper stride handling
        };
        
        size_t total_mat_size = sizeof(MatHeader) + data_size;

        LOGD("copy_to_queue: position={}, total_mat_size={}, capacity={}",
             position, total_mat_size, header_->capacity);

        if (total_mat_size > header_->capacity) {
            LOGD("Frame too large: total_mat_size={} exceeds capacity={}",
                 total_mat_size, header_->capacity);
            return false;
        }

        if (position + total_mat_size > header_->capacity) {
            LOGD("Wrapping buffer: position={} total_mat_size={}", position,
                 total_mat_size);
            position = 0;
        }

        // Write header first with proper memory ordering
        std::memcpy(buffer_ + position, &mat_header, sizeof(MatHeader));
        
        // Then copy the actual data
        std::memcpy(buffer_ + position + sizeof(MatHeader), continuous.data, data_size);
        
        // Ensure all memory writes are visible to other threads
        std::atomic_thread_fence(std::memory_order_release);

        return true;
    }

    // New method for zero-copy read
    bool read_from_queue_zero_copy(cv::Mat& result, size_t position) {
        // Get pointer to the header in shared memory
        MatHeader* header_ptr = reinterpret_cast<MatHeader*>(buffer_ + position);
        
        // Ensure memory is synchronized
        std::atomic_thread_fence(std::memory_order_acquire);
        
        // Validate header fields
        if (header_ptr->rows <= 0 || header_ptr->cols <= 0 ||
            header_ptr->data_size == 0 || 
            header_ptr->data_size > header_->capacity)
            return false;
            
        // Create a Mat header that points directly to shared memory data
        // This is zero-copy - we're just creating a view of the existing data
        uint8_t* data_ptr = buffer_ + position + sizeof(MatHeader);
        
        // In this simplified version, we don't increment the reference count.
        // For streaming applications, using a lock in serveLatestFrame is sufficient
        // to ensure the Mat is not overwritten while it's being processed.
        
        // Direct zero-copy by creating a Mat that references the shared memory
        result = cv::Mat(header_ptr->rows, header_ptr->cols, header_ptr->type, 
                         data_ptr, header_ptr->step);
        
        // For debugging
        LOGD("Created zero-copy Mat from shared memory at position {}", position);
        
        return true;
    }
    
    // Original method (kept for backward compatibility)
    bool read_from_queue(cv::Mat& result, size_t position) {
        MatHeader mat_header;
        std::atomic_thread_fence(std::memory_order_acquire);
        std::memcpy(&mat_header, buffer_ + position, sizeof(MatHeader));

        if (mat_header.rows <= 0 || mat_header.cols <= 0 ||
            mat_header.data_size == 0 ||
            mat_header.data_size > header_->capacity)
            return false;

        // Create a deep copy
        result.create(mat_header.rows, mat_header.cols, mat_header.type);
        std::memcpy(result.data, buffer_ + position + sizeof(MatHeader),
                    mat_header.data_size);
        return true;
    }

    size_t get_entry_size(size_t position) {
        MatHeader mat_header;
        std::atomic_thread_fence(std::memory_order_acquire);
        std::memcpy(&mat_header, buffer_ + position, sizeof(MatHeader));

        if (mat_header.rows <= 0 || mat_header.cols <= 0 ||
            mat_header.data_size == 0 ||
            mat_header.data_size > header_->capacity)
            return 0;

        return sizeof(MatHeader) + mat_header.data_size;
    }

   public:
    static MatQueue Create(const std::string& queue_name, size_t frame_count) {
        MatQueue queue;
        queue.name_ = queue_name;
        size_t page_size = sysconf(_SC_PAGE_SIZE);
        size_t buffer_size = frame_count * (MAX_FRAME_SIZE + sizeof(MatHeader));
        queue.total_size_ =
            sizeof(QueueHeader) +
            ((buffer_size + page_size - 1) / page_size) * page_size;

        queue.fd_ = shm_open(queue_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (queue.fd_ == -1) return queue;

        if (ftruncate(queue.fd_, queue.total_size_) == -1) {
            shm_unlink(queue_name.c_str());
            return queue;
        }

        queue.mapped_memory_ =
            mmap(NULL, queue.total_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                 queue.fd_, 0);
        if (queue.mapped_memory_ == MAP_FAILED) {
            shm_unlink(queue_name.c_str());
            return queue;
        }

        queue.header_ = static_cast<QueueHeader*>(queue.mapped_memory_);
        queue.buffer_ =
            static_cast<uint8_t*>(queue.mapped_memory_) + sizeof(QueueHeader);
        new (queue.header_) QueueHeader{0, 0, false, buffer_size};
        return queue;
    }

    // Helper method for zero-copy operations
    bool is_zero_copy_supported() const {
        return true;  // All MatQueue instances support zero-copy
    }
    
    // Enhanced try_pop with zero-copy support
    bool try_pop(cv::Mat& result, bool zero_copy) {
        if (!is_valid()) return false;

        // Use relaxed memory ordering for initial checks to reduce overhead
        size_t read_pos = header_->read_pos.load(std::memory_order_relaxed);
        size_t write_pos = header_->write_pos.load(std::memory_order_relaxed);
        bool was_overwritten = header_->was_overwritten.load(std::memory_order_relaxed);
        
        // Quick check if queue is empty
        if (read_pos == write_pos && !was_overwritten) return false;
        if (read_pos >= header_->capacity) read_pos = 0;

        // Acquire fence only if we're actually going to read data
        std::atomic_thread_fence(std::memory_order_acquire);
        size_t entry_size = get_entry_size(read_pos);
        if (entry_size == 0 || entry_size > header_->capacity) return false;

        size_t remaining_space = header_->capacity - read_pos;
        if (remaining_space < entry_size) read_pos = 0;
        
        // Use zero-copy or regular read based on parameter
        bool success = zero_copy ? 
            read_from_queue_zero_copy(result, read_pos) : 
            read_from_queue(result, read_pos);
            
        if (!success || result.empty()) return false;

        size_t next_pos = (read_pos + entry_size) % header_->capacity;
        // Use release ordering for updating the read position
        header_->read_pos.store(next_pos, std::memory_order_release);

        return true;
    }
    
    // Use a different name for the zero-copy version to avoid ambiguity
    bool try_pop_zero_copy(cv::Mat& result) {
        return try_pop(result, true); // Use zero-copy
    }
    
    // Original method for backward compatibility
    bool try_pop(cv::Mat& result) {
        return try_pop(result, false); // Use regular copy for compatibility
    }

    static MatQueue Open(const std::string& queue_name) {
        MatQueue queue;
        queue.fd_ = shm_open(queue_name.c_str(), O_RDWR, 0666);
        if (queue.fd_ == -1) return queue;

        struct stat sb;
        if (fstat(queue.fd_, &sb) == -1) return queue;
        queue.total_size_ = sb.st_size;

        queue.mapped_memory_ =
            mmap(NULL, queue.total_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                 queue.fd_, 0);
        if (queue.mapped_memory_ == MAP_FAILED) return queue;

        queue.header_ = static_cast<QueueHeader*>(queue.mapped_memory_);
        queue.buffer_ =
            static_cast<uint8_t*>(queue.mapped_memory_) + sizeof(QueueHeader);
        return queue;
    }

    static void Close(const std::string& queue_name) {
        shm_unlink(queue_name.c_str());
    }

    // Check if we can share memory between threads, avoiding unnecessary copying
    bool can_share_memory() const {
        // This is only used for optimization purposes
        return true;
    }
    
    bool try_push(const cv::Mat& mat) {
        if (!is_valid() || mat.empty()) return false;

        // Calculate required space for the mat
        size_t required_space = sizeof(MatHeader) + mat.total() * mat.elemSize();
        
        // Ensure we have enough space in the queue
        if (required_space > header_->capacity) {
            LOGW("Frame too large: required_space={} exceeds capacity={}", 
                 required_space, header_->capacity);
            return false;
        }
        
        // Read positions with relaxed memory ordering for initial calculation
        size_t write_pos = header_->write_pos.load(std::memory_order_relaxed);
        size_t read_pos = header_->read_pos.load(std::memory_order_relaxed);
        
        // Calculate available space
        size_t available_space = (write_pos >= read_pos)
            ? (header_->capacity - (write_pos - read_pos))
            : (read_pos - write_pos);

        // If insufficient space, evict old entries
        if (required_space >= available_space) {
            // Acquire memory order for eviction operations
            read_pos = header_->read_pos.load(std::memory_order_acquire);
            while (required_space > available_space) {
                size_t entry_size = get_entry_size(read_pos);
                if (entry_size == 0) break;

                read_pos = (read_pos + entry_size) % header_->capacity;
                available_space += entry_size;
            }
            header_->read_pos.store(read_pos, std::memory_order_release);
            header_->was_overwritten.store(true, std::memory_order_release);
        }

        // Handle buffer wrap-around
        size_t remaining_space = header_->capacity - write_pos;
        if (remaining_space < required_space) {
            write_pos = 0;
        }

        // Copy the frame to the queue
        if (!copy_to_queue(mat, write_pos)) return false;

        // Update write position with release ordering
        size_t next_pos = (write_pos + required_space) % header_->capacity;
        header_->write_pos.store(next_pos, std::memory_order_release);

        LOGD("pushed, write={}, read={}, was_overwritten={}", write_pos,
             read_pos,
             header_->was_overwritten.load(std::memory_order_acquire));
        return true;
    }

    bool is_valid() const {
        return mapped_memory_ && mapped_memory_ != MAP_FAILED && header_;
    }

    size_t size() const { return total_size_; }
};
}  // namespace pallas
