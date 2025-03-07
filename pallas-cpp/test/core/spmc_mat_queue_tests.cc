#include <core/logger.h>
#include <gtest/gtest.h>

#include <chrono>
#include <opencv2/core.hpp>
#include <thread>

#include "service/spmc_mat_queue.h"

namespace pallas {

class SimpleMatQueueTests : public testing::Test {
   protected:
    using Queue = MultiConsumerMatQueue<300>;

    void SetUp() override {
        // Clean up any leftover shared memory
        Queue::Close("simple_test");

        // Create a fresh queue for testing
        queue_ = Queue::Create("simple_test", 5);

        frames_.reserve(5);
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(0, 0, 0)));
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(1, 0, 0)));
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(2, 0, 0)));
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(3, 0, 0)));
        frames_.push_back(cv::Mat(10, 10, CV_8UC3, cv::Scalar(4, 0, 0)));
    }

    void TearDown() override { Queue::Close("simple_test"); }

    Queue queue_;
    std::vector<cv::Mat> frames_;
};

TEST_F(SimpleMatQueueTests, RegisterThenPush) {
    // First, make sure we have a valid queue
    EXPECT_TRUE(queue_.is_valid());

    // Register the consumer
    int consumer_id = queue_.register_consumer();
    EXPECT_GE(consumer_id, 0) << "Failed to register consumer";
    EXPECT_TRUE(queue_.is_consumer());

    // Push frames
    for (size_t i = 0; i < frames_.size(); ++i) {
        EXPECT_TRUE(queue_.try_push(frames_[i]))
            << "Failed to push frame " << i;
    }

    // Now read frames - consumer should see all frames pushed after
    // registration
    for (size_t i = 0; i < frames_.size(); ++i) {
        cv::Mat popped;

        bool pop_result = queue_.try_pop(popped);
        EXPECT_TRUE(pop_result) << "Failed to pop frame " << i;

        if (pop_result) {
            double min, max;
            cv::minMaxIdx(popped, &min, &max);
            EXPECT_DOUBLE_EQ(0.0, min);
            EXPECT_DOUBLE_EQ(static_cast<double>(i), max)
                << "Frame " << i << " has incorrect content";
        }
    }

    // Verify there are no more frames to read
    cv::Mat extra;
    EXPECT_FALSE(queue_.try_pop(extra)) << "Got unexpected extra frame";

    // Unregister consumer
    EXPECT_TRUE(queue_.unregister_consumer());
    EXPECT_FALSE(queue_.is_consumer());
}

TEST_F(SimpleMatQueueTests, PushThenRegister) {
    // First, make sure we have a valid queue
    EXPECT_TRUE(queue_.is_valid());

    // Push frames before registering any consumer
    for (size_t i = 0; i < frames_.size(); ++i) {
        EXPECT_TRUE(queue_.try_push(frames_[i]))
            << "Failed to push frame " << i;
    }

    // Now register a consumer
    int consumer_id = queue_.register_consumer();
    EXPECT_GE(consumer_id, 0) << "Failed to register consumer";

    // This consumer shouldn't see any of the previously pushed frames
    // because its read position will be set to the current write position
    cv::Mat popped;
    EXPECT_FALSE(queue_.try_pop(popped))
        << "Consumer incorrectly read a frame pushed before registration";

    // Push one more frame that the consumer should see
    cv::Mat new_frame(10, 10, CV_8UC3, cv::Scalar(10, 0, 0));
    EXPECT_TRUE(queue_.try_push(new_frame));

    // Now consumer should be able to read this frame
    bool pop_result = queue_.try_pop(popped);
    EXPECT_TRUE(pop_result) << "Failed to pop frame pushed after registration";

    if (pop_result) {
        double min, max;
        cv::minMaxIdx(popped, &min, &max);
        EXPECT_DOUBLE_EQ(0.0, min);
        EXPECT_DOUBLE_EQ(10.0, max) << "Frame has incorrect content";
    }

    // Unregister consumer
    EXPECT_TRUE(queue_.unregister_consumer());
}

TEST_F(SimpleMatQueueTests, TwoConsumers) {
    // First, make sure we have a valid queue
    EXPECT_TRUE(queue_.is_valid());

    // Register first consumer
    int consumer1_id = queue_.register_consumer();
    EXPECT_GE(consumer1_id, 0) << "Failed to register first consumer";

    // Push first set of frames
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(queue_.try_push(frames_[i]))
            << "Failed to push frame " << i;
    }

    // Register second consumer
    auto consumer2_queue = Queue::Open("simple_test");
    int consumer2_id = consumer2_queue.register_consumer();
    EXPECT_GE(consumer2_id, 0) << "Failed to register second consumer";

    // Push more frames
    for (size_t i = 2; i < frames_.size(); ++i) {
        EXPECT_TRUE(queue_.try_push(frames_[i]))
            << "Failed to push frame " << i;
    }

    // First consumer should see all frames
    for (size_t i = 0; i < frames_.size(); ++i) {
        cv::Mat popped;
        bool pop_result = queue_.try_pop(popped);
        EXPECT_TRUE(pop_result) << "First consumer failed to pop frame " << i;

        if (pop_result) {
            double min, max;
            cv::minMaxIdx(popped, &min, &max);
            EXPECT_DOUBLE_EQ(0.0, min);
            EXPECT_DOUBLE_EQ(static_cast<double>(i), max)
                << "Frame " << i << " has incorrect content for first consumer";
        }
    }

    // Second consumer should only see frames pushed after it registered
    for (size_t i = 2; i < frames_.size(); ++i) {
        cv::Mat popped;
        bool pop_result = consumer2_queue.try_pop(popped);
        EXPECT_TRUE(pop_result) << "Second consumer failed to pop frame " << i;

        if (pop_result) {
            double min, max;
            cv::minMaxIdx(popped, &min, &max);
            EXPECT_DOUBLE_EQ(0.0, min);
            EXPECT_DOUBLE_EQ(static_cast<double>(i), max)
                << "Frame " << i
                << " has incorrect content for second consumer";
        }
    }

    // Verify there are no more frames to read
    cv::Mat extra;
    EXPECT_FALSE(queue_.try_pop(extra))
        << "First consumer got unexpected extra frame";
    EXPECT_FALSE(consumer2_queue.try_pop(extra))
        << "Second consumer got unexpected extra frame";

    // Unregister consumers
    EXPECT_TRUE(queue_.unregister_consumer());
    EXPECT_TRUE(consumer2_queue.unregister_consumer());
}
}  // namespace pallas
