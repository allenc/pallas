#pragma once
#include <core/logger.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstring>
#include <array>
#include <opencv2/core.hpp>
#include <string>

namespace pallas {

constexpr size_t MAX_CONSUMERS = 8;

template <size_t MAX_FRAME_SIZE>
class MultiConsumerMatQueue {
   private:
    struct alignas(64) MatHeader {
        int rows;
        int cols;
        int type;
        size_t data_size;
    };

    struct alignas(64) QueueHeader {
        std::atomic<size_t> write_pos;
        std::array<std::atomic<size_t>, MAX_CONSUMERS> read_positions;
        std::array<std::atomic<bool>, MAX_CONSUMERS> was_overwritten;
        std::array<std::atomic<bool>, MAX_CONSUMERS> consumer_active;
        std::atomic<size_t> min_read_pos;
        size_t capacity;
        std::atomic<int> registration_lock;
    };

    void* mapped_memory_ = nullptr;
    QueueHeader* header_ = nullptr;
    uint8_t* buffer_ = nullptr;
    int fd_ = -1;
    std::string name_;
    size_t total_size_ = 0;
    int consumer_id_ = -1;

    bool copy_to_queue(const cv::Mat& mat, size_t position) {
        cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();
        size_t data_size = continuous.total() * continuous.elemSize();
        MatHeader mat_header{continuous.rows, continuous.cols,
                            continuous.type(), data_size};
        size_t total_mat_size = sizeof(MatHeader) + data_size;

        if (total_mat_size > header_->capacity) return false;

        if (position + total_mat_size > header_->capacity) {
            position = 0;
        }

        std::memcpy(buffer_ + position, &mat_header, sizeof(MatHeader));
        std::memcpy(buffer_ + position + sizeof(MatHeader), continuous.data, data_size);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        return true;
    }

    bool read_from_queue(cv::Mat& result, size_t position) {
        MatHeader mat_header;
        std::atomic_thread_fence(std::memory_order_acquire);
        std::memcpy(&mat_header, buffer_ + position, sizeof(MatHeader));

        if (mat_header.rows <= 0 || mat_header.cols <= 0 ||
            mat_header.data_size == 0 ||
            mat_header.data_size > header_->capacity)
            return false;

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
    
    void update_min_read_pos() {
        size_t min_pos = header_->write_pos.load(std::memory_order_acquire);
        bool has_consumers = false;
        
        for (size_t i = 0; i < MAX_CONSUMERS; i++) {
            if (header_->consumer_active[i].load(std::memory_order_acquire)) {
                has_consumers = true;
                size_t pos = header_->read_positions[i].load(std::memory_order_acquire);
                if (pos < min_pos) min_pos = pos;
            }
        }
        
        if (has_consumers) {
            header_->min_read_pos.store(min_pos, std::memory_order_release);
        }
    }
    
    void acquire_registration_lock() {
        int expected = 0;
        while (!header_->registration_lock.compare_exchange_weak(
            expected, 1, std::memory_order_acquire)) {
            expected = 0;
            std::this_thread::yield();
        }
    }
    
    void release_registration_lock() {
        header_->registration_lock.store(0, std::memory_order_release);
    }

   public:
    static MultiConsumerMatQueue Create(const std::string& queue_name, size_t frame_count) {
        // First close any existing queue with this name
        shm_unlink(queue_name.c_str());
        
        MultiConsumerMatQueue queue;
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
        
        // Initialize header with atomic values
        new (queue.header_) QueueHeader();
        queue.header_->write_pos.store(0, std::memory_order_relaxed);
        queue.header_->min_read_pos.store(0, std::memory_order_relaxed);
        queue.header_->capacity = buffer_size;
        queue.header_->registration_lock.store(0, std::memory_order_relaxed);
        
        for (size_t i = 0; i < MAX_CONSUMERS; i++) {
            queue.header_->read_positions[i].store(0, std::memory_order_relaxed);
            queue.header_->was_overwritten[i].store(false, std::memory_order_relaxed);
            queue.header_->consumer_active[i].store(false, std::memory_order_relaxed);
        }
        
        return queue;
    }
    
    int register_consumer() {
        if (!is_valid()) return -1;
        
        acquire_registration_lock();
        
        int assigned_id = -1;
        for (int i = 0; i < MAX_CONSUMERS; i++) {
            if (!header_->consumer_active[i].load(std::memory_order_acquire)) {
                // Set read position to current write position
                // This ensures the consumer only sees frames pushed after registration
                size_t initial_pos = header_->write_pos.load(std::memory_order_acquire);
                
                header_->consumer_active[i].store(true, std::memory_order_release);
                header_->read_positions[i].store(initial_pos, std::memory_order_release);
                header_->was_overwritten[i].store(false, std::memory_order_release);
                assigned_id = i;
                break;
            }
        }
        
        if (assigned_id >= 0) {
            consumer_id_ = assigned_id;
            update_min_read_pos();
        }
        
        release_registration_lock();
        return assigned_id;
    }
    
    bool unregister_consumer() {
        if (!is_valid() || consumer_id_ < 0 || consumer_id_ >= MAX_CONSUMERS) 
            return false;
            
        acquire_registration_lock();
        
        header_->consumer_active[consumer_id_].store(false, std::memory_order_release);
        update_min_read_pos();
        consumer_id_ = -1;
        
        release_registration_lock();
        return true;
    }

    bool try_pop(cv::Mat& result) {
        if (!is_valid() || consumer_id_ < 0 || consumer_id_ >= MAX_CONSUMERS) 
            return false;
            
        if (!header_->consumer_active[consumer_id_].load(std::memory_order_acquire))
            return false;
            
        size_t read_pos = header_->read_positions[consumer_id_].load(std::memory_order_acquire);
        size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
        bool was_overwritten = header_->was_overwritten[consumer_id_].load(std::memory_order_acquire);
        
        // Nothing to read if read position equals write position and no overwrite occurred
        if (read_pos == write_pos && !was_overwritten) return false;
        
        if (read_pos >= header_->capacity) read_pos = 0;
        
        std::atomic_thread_fence(std::memory_order_acquire);
        size_t entry_size = get_entry_size(read_pos);
        if (entry_size == 0 || entry_size > header_->capacity) {
            // If we can't get a valid entry size, move to the write position
            header_->read_positions[consumer_id_].store(write_pos, std::memory_order_release);
            header_->was_overwritten[consumer_id_].store(false, std::memory_order_release);
            return false;
        }

        size_t remaining_space = header_->capacity - read_pos;
        if (remaining_space < entry_size) {
            read_pos = 0;
            entry_size = get_entry_size(read_pos);
            if (entry_size == 0 || entry_size > header_->capacity) {
                header_->read_positions[consumer_id_].store(write_pos, std::memory_order_release);
                header_->was_overwritten[consumer_id_].store(false, std::memory_order_release);
                return false;
            }
        }
        
        if (!read_from_queue(result, read_pos)) {
            header_->read_positions[consumer_id_].store(write_pos, std::memory_order_release);
            header_->was_overwritten[consumer_id_].store(false, std::memory_order_release);
            return false;
        }
        
        if (result.empty()) {
            header_->read_positions[consumer_id_].store(write_pos, std::memory_order_release);
            header_->was_overwritten[consumer_id_].store(false, std::memory_order_release);
            return false;
        }

        // Reset overwritten flag after successful read
        if (was_overwritten) {
            header_->was_overwritten[consumer_id_].store(false, std::memory_order_release);
        }
        
        // Update read position
        size_t next_pos = (read_pos + entry_size) % header_->capacity;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        header_->read_positions[consumer_id_].store(next_pos, std::memory_order_release);
        
        // Occasionally update min read position to prevent excessive contention
        if (rand() % 5 == 0) {
            update_min_read_pos();
        }

        return true;
    }

    static MultiConsumerMatQueue Open(const std::string& queue_name) {
        MultiConsumerMatQueue queue;
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

    bool try_push(const cv::Mat& mat) {
        if (!is_valid() || mat.empty()) return false;

        size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
        size_t min_read_pos = header_->min_read_pos.load(std::memory_order_acquire);
        size_t required_space = sizeof(MatHeader) + mat.total() * mat.elemSize();

        // Calculate available space
        size_t available_space =
            (write_pos >= min_read_pos)
                ? (header_->capacity - (write_pos - min_read_pos))
                : (min_read_pos - write_pos);

        // If not enough space, buffer will be overwritten
        bool overwriting = required_space >= available_space;
        
        // Handle buffer wrap-around
        bool wrapping = false;
        size_t remaining_space = header_->capacity - write_pos;
        if (remaining_space < required_space) {
            write_pos = 0;
            wrapping = true;
        }

        if (!copy_to_queue(mat, write_pos)) return false;

        // Update write position
        size_t next_pos = (write_pos + required_space) % header_->capacity;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        // Set overwritten flag for all active consumers if wrapping or overwriting
        if (wrapping || overwriting) {
            for (size_t i = 0; i < MAX_CONSUMERS; i++) {
                if (header_->consumer_active[i].load(std::memory_order_acquire)) {
                    header_->was_overwritten[i].store(true, std::memory_order_release);
                }
            }
        }
        
        header_->write_pos.store(next_pos, std::memory_order_release);
        return true;
    }

    bool is_valid() const {
        return mapped_memory_ && mapped_memory_ != MAP_FAILED && header_;
    }

    size_t size() const { return total_size_; }
    
    int get_consumer_id() const { return consumer_id_; }
    
    bool is_consumer() const { return consumer_id_ >= 0; }
};
}  // namespace pallas
