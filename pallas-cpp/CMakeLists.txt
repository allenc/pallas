cmake_minimum_required(VERSION 3.16.0)
project(pallas-cpp)

set(DEV_MODULE Development)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

message("${PROJECT_SOURCE_DIR}/.cmake")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/.cmake" ${CMAKE_MODULE_PATH})

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(onnxruntime REQUIRED)

# Nix version of spdlog
add_compile_definitions(SPDLOG_FMT_EXTERNAL)

# Using Mongoose for HTTP server
set(MONGOOSE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/mongoose_include")
file(MAKE_DIRECTORY ${MONGOOSE_INCLUDE_DIR})

if(NOT EXISTS "${MONGOOSE_INCLUDE_DIR}/mongoose.h" OR NOT EXISTS "${MONGOOSE_INCLUDE_DIR}/mongoose.c")
    message(STATUS "Downloading Mongoose library")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/cesanta/mongoose/master/mongoose.h"
        "${MONGOOSE_INCLUDE_DIR}/mongoose.h"
        SHOW_PROGRESS
        TLS_VERIFY ON
    )
    file(DOWNLOAD
        "https://raw.githubusercontent.com/cesanta/mongoose/master/mongoose.c"
        "${MONGOOSE_INCLUDE_DIR}/mongoose.c"
        SHOW_PROGRESS
        TLS_VERIFY ON
    )
endif()

# Download JSON library if not found
set(JSON_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/json_include")
file(MAKE_DIRECTORY ${JSON_INCLUDE_DIR}/nlohmann)

if(NOT EXISTS "${JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(STATUS "Downloading nlohmann/json.hpp")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
        "${JSON_INCLUDE_DIR}/nlohmann/json.hpp"
        SHOW_PROGRESS
        TLS_VERIFY ON
    )
endif()

message(STATUS "Using Mongoose from: ${MONGOOSE_INCLUDE_DIR}")
message(STATUS "Using nlohmann_json from: ${JSON_INCLUDE_DIR}")

# -- Core Library --
add_library(core STATIC
  src/core/logger.cc
  src/core/service.cc
  src/core/timer.cc
)
target_include_directories(core PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src
)
target_link_libraries(core PRIVATE spdlog::spdlog)

# -- Vision Library --
add_library(vision STATIC
  src/vision/detection.cc
  src/vision/geometry.cc
  src/vision/sam.cc      
  src/vision/yolo.cc
  src/vision/yolo_utils.cc          
)

target_include_directories(vision PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src
)
target_link_libraries(vision PRIVATE core onnxruntime ${OpenCV_LIBS})

# -- Vision Python Bindings --
nanobind_add_module(pallas_py
  bindings/geometry.cc
)

target_link_libraries(pallas_py PRIVATE core vision)

# -- Executables: Services --
add_executable(psystreamd
  process/psystreamd.cc
  src/service/inference_service.cc
)
target_include_directories(psystreamd PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src
)
target_link_libraries(psystreamd PUBLIC core vision)
  
add_executable(starburstd
  process/starburstd.cc
  src/service/camera_service.cc  
)
target_include_directories(starburstd PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src
)
target_link_libraries(starburstd PUBLIC core ${OpenCV_LIBS})

add_executable(starforged
  process/starforged.cc
  src/service/viewer_service.cc  
)
target_include_directories(starforged PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src  
)
target_link_libraries(starforged PUBLIC core PRIVATE ${OpenCV_LIBS})    

# New stream service with HTTP server
add_executable(streamd
  process/streamd.cc
  src/service/stream_service.cc
  ${MONGOOSE_INCLUDE_DIR}/mongoose.c
)
target_include_directories(streamd PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${MONGOOSE_INCLUDE_DIR}
  ${JSON_INCLUDE_DIR}
)
target_link_libraries(streamd PUBLIC
  core
  ${OpenCV_LIBS}
  pthread
)
# Define MG_ENABLE_OPENSSL=0 to disable OpenSSL
target_compile_definitions(streamd PRIVATE 
    MG_ENABLE_OPENSSL=0
)

# -- Executables: Tests --

add_executable(unit-tests
    test/main_test.cc  
    test/core/mat_queue_tests.cc
    test/core/spmc_mat_queue_tests.cc
    test/vision/geometry_tests.cc    
    test/vision/sam_tests.cc
    test/vision/yolo_tests.cc        
)    
target_include_directories(unit-tests PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CMAKE_CURRENT_LIST_DIR}/test  
)
target_link_libraries(unit-tests PUBLIC core vision gtest)  


# -- Install --
install(TARGETS core LIBRARY DESTINATION lib)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/core" DESTINATION include FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/frontend/" DESTINATION share/pallas/frontend)

# -- Formatting --
function(add_clang_format_target)
    find_program(CLANG_FORMAT "clang-format")
    if(NOT CLANG_FORMAT)
        message(WARNING "clang-format not found, formatting target will not be available")
        return()
    endif()

    # Recursively find all source files
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/include/*.cc
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/src/*.cc
        ${CMAKE_SOURCE_DIR}/src/*.h
        ${CMAKE_SOURCE_DIR}/test/*.cc
        ${CMAKE_SOURCE_DIR}/test/*.h
    )

    # Print each file found
    foreach(file ${ALL_SOURCE_FILES})
        message(STATUS "Found file: ${file}")
    endforeach()

    # Add the custom target to format the code
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${ALL_SOURCE_FILES}
        COMMENT "Running clang-format on all source files"
        VERBATIM
    )
endfunction()

# Call the function to add the format target
add_clang_format_target()
