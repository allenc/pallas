# Pallas Development Guide

## Build & Test Commands
- Build: `cmake -B build && cmake --build build`
- Format: `make format` (uses clang-format)
- Run all tests: `./build/unit-tests`
- Run specific test: `./build/unit-tests --gtest_filter=TestSuiteName.TestName`
- Common test suites: GeometryTest, YoloTest, SamTest, MatQueueTest, SpmcMatQueueTest

## Code Style
- **C++ Standard**: C++23
- **Formatting**: 4 space indentation, clang-format enforced
- **Namespaces**: lowercase (`pallas`)
- **Classes/Structs**: PascalCase (`CameraService`)
- **Methods/Functions**: camelCase for methods, snake_case for free functions
- **Constants/Macros**: UPPERCASE
- **Member Variables**: snake_case with trailing underscore (`running_`)
- **Imports**: Standard libs → Third-party libs → Project headers (quotes)

## Error Handling
- Use `std::expected<T, E>` for service functions that may fail
- Logging: `LOGI`/`LOGD`/`LOGE` for info/debug/error levels
- `LOGA` for assertions with logging