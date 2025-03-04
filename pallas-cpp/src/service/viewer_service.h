#pragma once

#include <core/service.h>

#include <expected>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mat_queue.h"

namespace pallas {

struct ViewerServiceConfig {
    ServiceConfig base;
    std::vector<std::string> shared_memory_names;
};

class ViewerService : public Service {
   public:
    using Queue = MatQueue<2764800>;

    ViewerService(ViewerServiceConfig config);
    bool start() override;
    void stop() override;

   protected:
    std::expected<void, std::string> tick() override;

   private:
    std::vector<std::string> shared_memory_names_;
    std::unordered_map<std::string, std::unique_ptr<Queue>> queue_by_name_;
};
}  // namespace pallas
