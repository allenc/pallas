#include <core/shm_spsc.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

const std::string SHARED_MEMORY_NAME = "queue";
const std::size_t QUEUE_CAPACITY = 500;

namespace pallas {

void producer() {
  // Ensure any previously existing shared memory is cleaned up
  SharedMemorySPSCQueue<int>::Close(SHARED_MEMORY_NAME);

  auto queue =
      SharedMemorySPSCQueue<int>::Create(SHARED_MEMORY_NAME, QUEUE_CAPACITY);
  for (int i = 0; i < 10000; ++i) {
    while (!queue.try_push(i)) {
      std::this_thread::yield(); // Busy wait if the queue is full
    }
    std::cout << "Produced: " << i << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void consumer() {
  // Wait to ensure the producer initializes shared memory first
  std::this_thread::sleep_for(std::chrono::seconds(1));

  auto queue = SharedMemorySPSCQueue<int>::Open(SHARED_MEMORY_NAME);
  for (int i = 0; i < 10000; ++i) {
    int value;
    while (!queue.try_pop(value)) {
      std::this_thread::yield(); // Busy wait if the queue is empty
    }
    std::cout << "Consumed: " << value << std::endl;
  }
  SharedMemorySPSCQueue<int>::Close(SHARED_MEMORY_NAME);
}
	
} // namespace pallas	

int main() {
  pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "Fork failed" << std::endl;
    return EXIT_FAILURE;
  } else if (pid == 0) {
	  pallas::consumer(); // Child process runs consumer
  } else {
	  pallas::producer();               // Parent process runs producer
    waitpid(pid, nullptr, 0); // Wait for child process to finish
  }
  return EXIT_SUCCESS;
}
