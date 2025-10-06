cmake_minimum_required(VERSION 3.14)
project(llama_api_server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Include llama.cpp as a sub-directory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp)

# 2. Download cpp-httplib if not present
include(FetchContent)
FetchContent_Declare(
    httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.14.3.tar.gz
)
FetchContent_MakeAvailable(httplib)

# 3. Find nlohmann_json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# 4. Find Poppler library for PDF processing
find_package(PkgConfig REQUIRED)
pkg_check_modules(POPPLER REQUIRED poppler-cpp)

# 5. Create executable for CV detection server (IMAGE MODE)
add_executable(llama_api_server_cv llama_api_server_cv_detection.cpp)

# 6. Link libraries for CV detection (no llama library needed - uses CLI)
target_link_libraries(llama_api_server_cv
    PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ${POPPLER_LIBRARIES}
)

# 7. Include directories for CV detection
target_include_directories(llama_api_server_cv
    PRIVATE
    ${POPPLER_INCLUDE_DIRS}
)

# 8. Add compile options
target_compile_options(llama_api_server_cv
    PRIVATE
    ${POPPLER_CFLAGS_OTHER}
)

# 9. (Optional) Create the original persona server as well
add_executable(llama_api_server llama_api_server.cpp)

target_link_libraries(llama_api_server
    PRIVATE
    llama
    httplib::httplib
    nlohmann_json::nlohmann_json
)

target_include_directories(llama_api_server
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/ggml/include
)

# 10. Create uploads and temp directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/uploads)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/uploads/temp)

# Also create in source directory for easier testing
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/uploads)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/uploads/temp)

# 11. Print build information
message(STATUS "Building llama API servers:")
message(STATUS "  - CV Detection Server (IMAGE MODE): llama_api_server_cv")
message(STATUS "  - Persona Server: llama_api_server")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  Poppler found: ${POPPLER_FOUND}")
message(STATUS "  Poppler include dirs: ${POPPLER_INCLUDE_DIRS}")
message(STATUS "  Poppler libraries: ${POPPLER_LIBRARIES}")
message(STATUS "")
message(STATUS "Directory structure:")
message(STATUS "  Build directory: ${CMAKE_BINARY_DIR}")
message(STATUS "  Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Uploads will be in: ${CMAKE_CURRENT_SOURCE_DIR}/uploads")