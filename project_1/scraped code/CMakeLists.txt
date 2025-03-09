cmake_minimum_required(VERSION 3.10)
project(Data CXX)

# Set C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Find and link zlib (required for cnpy)
find_package(ZLIB REQUIRED)

# Set optimization flag for Release mode
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add source files
add_executable(main main.cpp cnpy.cpp)

# Include directories
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link OpenMP and zlib
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX ZLIB::ZLIB)
