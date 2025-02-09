#pragma once

#include <core/mat_queue.h>
#include <core/result.h>
#include <core/service.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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
    Result<bool> tick() override;

   private:
    std::vector<std::string> shared_memory_names_;
    std::unordered_map<std::string, std::unique_ptr<Queue>> queue_by_name_;
};
}  // namespace pallas
