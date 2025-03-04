#include <core/logger.h>
#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "service/mat_queue.h"

namespace pallas {

class MatQueueTests : public testing::Test {
   protected:
    using Queue = MatQueue<300>;

    void SetUp() override {
        queue_ = Queue::Create("test", 5);

        frames_.reserve(5);
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(0, 0, 0)));
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(1, 0, 0)));
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(2, 0, 0)));
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(3, 0, 0)));
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(4, 0, 0)));
    }

    void TearDown() override { Queue::Close("test"); }

    Queue queue_;
    std::vector<cv::Mat> frames_;
};

TEST_F(MatQueueTests, SingleConsumer) {
    // Fill the queue.
    for (const auto& frame : frames_) {
        EXPECT_TRUE(queue_.try_push(frame));
    }

    // Add another frame to cause overflow.
    cv::Mat pushed(10, 10, CV_8UC3, cv::Scalar(6, 0, 0));
    EXPECT_TRUE(queue_.try_push(pushed));

    // The popped frame should be the added frame.
    cv::Mat popped;
    EXPECT_TRUE(queue_.try_pop(popped));
    EXPECT_EQ(0, cv::norm(pushed - popped));

    // The remainder frames should be preserved.
    for (std::size_t i = 1; i < frames_.size(); ++i) {
        cv::Mat popped;
        EXPECT_TRUE(queue_.try_pop(popped));
        double min, max;
        cv::minMaxIdx(popped, &min, &max);
        EXPECT_DOUBLE_EQ(0.0, min);
        EXPECT_DOUBLE_EQ(static_cast<double>(i), max);
    }
}

}  // namespace pallas
