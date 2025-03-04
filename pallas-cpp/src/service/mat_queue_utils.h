#pragma once

#include <fmt/format.h>

#include <expected>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mat_queue.h"

namespace pallas {
namespace utils {
template <typename Queue>
std::expected<std::unordered_map<std::string, std::unique_ptr<Queue>>,
              std::string>
open_verified_queues(const std::vector<std::string>& shared_memory_names) {
    std::unordered_map<std::string, std::unique_ptr<Queue>> queue_by_name;
    queue_by_name.reserve(shared_memory_names.size());
    for (const auto& name : shared_memory_names) {
        int fd = shm_open(name.c_str(), O_RDWR, 0);
        if (fd == -1) {
            // Invalid shared memory permissions.
            return std::unexpected(fmt::format(
                "Failed to initialize ViewerService with shared memory {}.",
                name));
        }
        close(fd);

        auto queue = Queue::Open(name);
        auto queue_ptr = std::make_unique<Queue>(std::move(queue));
        queue_by_name[name] = std::move(queue_ptr);
    }
    if (queue_by_name.empty()) {
        return std::unexpected(
            "Failed to initailize ViewerService with any shared memory "
            "queues.");
    }

    return queue_by_name;
}

}  // namespace utils
}  // namespace pallas
