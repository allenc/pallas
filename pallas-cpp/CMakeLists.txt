cmake_minimum_required(VERSION 3.16.0)
project(pallas-cpp)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

message("${PROJECT_SOURCE_DIR}/.cmake")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/.cmake" ${CMAKE_MODULE_PATH})

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(ftxui REQUIRED)

# Nix version of spdlog
add_compile_definitions(SPDLOG_FMT_EXTERNAL)

# -- Libraries --
add_library(core STATIC
  src/core/sam.cc    
  src/core/service.cc
  src/core/timer.cc
  src/core/yolo.cc
  src/core/yolo_utils.cc        
)

target_include_directories(core PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  # ${ONNXRUNTIME_ROOT_DIR}/include
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src
)
target_link_libraries(core PRIVATE onnxruntime ${OpenCV_LIBS} spdlog::spdlog)

# -- Executables: Services --
add_executable(psystreamd
  process/psystreamd.cc
  src/service/inference_service.cc        
)
target_include_directories(psystreamd PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src
  )
  target_link_libraries(psystreamd PUBLIC core)
  
add_executable(starburstd
  process/starburstd.cc
  src/service/camera_service.cc  
)
target_include_directories(starburstd PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src
  )
  target_link_libraries(starburstd PUBLIC core)

add_executable(starforged
  process/starforged.cc
  src/service/viewer_service.cc  
)
target_include_directories(starforged PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src  
  )
  target_link_libraries(starforged PUBLIC core
    PRIVATE ftxui::screen ftxui::dom ftxui::component)  

  # -- Executables: Tests --
  add_executable(core-tests
    test/core/mat_queue_tests.cc
    test/core/sam_tests.cc
    test/core/yolo_tests.cc        
    test/main_test.cc
)    
target_include_directories(core-tests PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/test  
  )
  target_link_libraries(core-tests PUBLIC core gtest)


# -- Install --
install(TARGETS core LIBRARY DESTINATION lib)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/core" DESTINATION include FILES_MATCHING PATTERN "*.h")

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
