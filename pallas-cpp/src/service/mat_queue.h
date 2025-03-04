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
        cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();
        size_t data_size = continuous.total() * continuous.elemSize();

        MatHeader mat_header{continuous.rows, continuous.cols,
                             continuous.type(), data_size};
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

        std::memcpy(buffer_ + position, &mat_header, sizeof(MatHeader));
        std::memcpy(buffer_ + position + sizeof(MatHeader), continuous.data,
                    data_size);
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

    bool try_pop(cv::Mat& result) {
        if (!is_valid()) return false;

        size_t read_pos = header_->read_pos.load(std::memory_order_acquire);
        size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
        bool was_overwritten =
            header_->was_overwritten.load(std::memory_order_acquire);
        if (read_pos == write_pos && !was_overwritten) return false;
        if (read_pos >= header_->capacity) read_pos = 0;

        std::atomic_thread_fence(std::memory_order_acquire);
        size_t entry_size = get_entry_size(read_pos);
        if (entry_size == 0 || entry_size > header_->capacity) return false;

        size_t remaining_space = header_->capacity - read_pos;
        if (remaining_space < entry_size) read_pos = 0;
        if (!read_from_queue(result, read_pos)) return false;
        if (result.empty()) return false;

        size_t next_pos = (read_pos + entry_size) % header_->capacity;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        header_->read_pos.store(next_pos, std::memory_order_release);

        return true;
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

    bool try_push(const cv::Mat& mat) {
        if (!is_valid() || mat.empty()) return false;

        size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
        size_t read_pos = header_->read_pos.load(std::memory_order_acquire);
        size_t required_space =
            sizeof(MatHeader) + mat.total() * mat.elemSize();

        size_t available_space =
            (write_pos >= read_pos)
                ? (header_->capacity - (write_pos - read_pos))
                : (read_pos - write_pos);

        if (required_space >= available_space) {
            while (required_space > available_space) {
                size_t entry_size = get_entry_size(read_pos);
                if (entry_size == 0) break;

                read_pos = (read_pos + entry_size) % header_->capacity;
                available_space += entry_size;
            }
            header_->read_pos.store(read_pos, std::memory_order_release);
            header_->was_overwritten.store(true, std::memory_order_release);
        }

        size_t remaining_space = header_->capacity - write_pos;
        if (remaining_space < required_space) {
            write_pos = 0;
        }

        if (!copy_to_queue(mat, write_pos)) return false;

        size_t next_pos = (write_pos + required_space) % header_->capacity;
        std::atomic_thread_fence(std::memory_order_seq_cst);
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
