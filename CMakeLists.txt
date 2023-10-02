cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
#cmake_minimum_required(VERSION 2.8)
project(sipm-massive CXX)
# Require C++17 in all targets
set(CMAKE_CXX_STANDARD 17)
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
# Include the fast-cpp-csv-parser headers
include_directories(${CMAKE_SOURCE_DIR}/external/rapidcsv/src)

target_include_directories(sipm-massive PRIVATE include)
target_link_libraries(sipm-massive
  PUBLIC yaml-cpp
  PUBLIC Boost::boost Boost::log Boost::program_options
  PUBLIC Boost::stacktrace_basic -rdynamic
  PUBLIC ROOT::Core ROOT::Hist ROOT::Tree
  ${ROOT_LIBRARIES}
)
add_executable(scan src/scan.cc)
target_include_directories(scan PRIVATE include)
target_link_libraries(scan  PRIVATE sipm-massive)

add_executable(print src/print.cc)
target_include_directories(print PRIVATE include)
target_link_libraries(print  PRIVATE sipm-massive)

add_executable(skim-signal src/skim-signal.cc)
target_include_directories(skim-signal PRIVATE include)
target_link_libraries(skim-signal  PRIVATE sipm-massive)

add_executable(store-waveform src/store-waveform.cc)
target_include_directories(store-waveform PRIVATE include)
target_link_libraries(store-waveform  PRIVATE sipm-massive)

add_executable(skim-dcr src/dcr.cc)
target_include_directories(skim-dcr PRIVATE include)
target_link_libraries(skim-dcr  PRIVATE sipm-massive)

add_executable(skim-ap src/ap.cc)
target_include_directories(skim-ap PRIVATE include)
target_link_libraries(skim-ap  PRIVATE sipm-massive)

add_executable(new_dcr src/new_dcr.cc)
target_include_directories(new_dcr PRIVATE include)
target_link_libraries(new_dcr  PRIVATE sipm-massive)

