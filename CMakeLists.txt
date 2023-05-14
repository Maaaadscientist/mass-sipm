cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
#cmake_minimum_required(VERSION 2.8)
project(sipm-massive CXX)
# Require C++17 in all targets
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -pedantic)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")

find_package(ROOT 6 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(
  Boost 1.72 REQUIRED
  COMPONENTS log program_options stacktrace_basic
)


add_library(sipm-massive SHARED
  src/Logger.cc
  src/Options.cc
)
target_include_directories(sipm-massive PRIVATE include)
target_link_libraries(sipm-massive
  PUBLIC yaml-cpp
  PUBLIC Boost::boost Boost::log Boost::program_options
  PUBLIC Boost::stacktrace_basic -rdynamic
  PUBLIC ROOT::Core ROOT::Hist ROOT::Tree
  #${ROOT_LIBRARIES}
)
add_executable(scan src/scan.cc)
target_include_directories(scan PRIVATE include)
target_link_libraries(scan  PRIVATE sipm-massive)

