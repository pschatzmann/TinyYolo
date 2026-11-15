cmake_minimum_required(VERSION 3.16)
project(TinyYolo VERSION 0.1.0 LANGUAGES C CXX)

# TinyYolo is header-only. These targets only build and test it on the desktop.
# The Arduino sketch build uses arduino-cli (see README.md).

option(TINYYOLO_BUILD_TESTS "Build the desktop test suite" ON)

find_package(Threads REQUIRED)

add_library(TinyYolo INTERFACE)
target_include_directories(TinyYolo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(TinyYolo INTERFACE cxx_std_11)
target_link_libraries(TinyYolo INTERFACE Threads::Threads)

# Desktop tools (see desktop/CMakeLists.txt).
add_subdirectory(desktop)

if(TINYYOLO_BUILD_TESTS)
  enable_testing()

  set(TINYYOLO_WARNINGS -Wall -Wextra -Wpedantic -Werror -Wformat=2)

  # Full test suite (exceptions and RTTI on, sanitizers optional).
  add_executable(tinyyolo_tests tests/host/test_tinyyolo.cpp tests/host/second_tu.cpp)
  target_link_libraries(tinyyolo_tests PRIVATE TinyYolo)
  target_compile_options(tinyyolo_tests PRIVATE ${TINYYOLO_WARNINGS})
  set_target_properties(tinyyolo_tests PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

  option(TINYYOLO_SANITIZE "Build tests with AddressSanitizer and UBSan" OFF)
  if(TINYYOLO_SANITIZE)
    target_compile_options(tinyyolo_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(tinyyolo_tests PRIVATE -fsanitize=address,undefined)
  endif()

  add_test(NAME tinyyolo_unit_tests COMMAND tinyyolo_tests)

  # The single-file model format: the round trip and the reader's checks.
  add_executable(tinyyolo_model_file_tests tests/host/test_model_file.cpp)
  target_link_libraries(tinyyolo_model_file_tests PRIVATE TinyYolo)
  target_compile_options(tinyyolo_model_file_tests PRIVATE ${TINYYOLO_WARNINGS})
  set_target_properties(tinyyolo_model_file_tests PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
  add_test(NAME tinyyolo_model_file_tests COMMAND tinyyolo_model_file_tests)

  # Strict compile: the flags Arduino cores use (no exceptions, no RTTI, C++11).
  # Also checks the library compiles as C++17 (ESP32 / RP2040 / nRF52 cores).
  foreach(std 11 17)
    add_library(tinyyolo_strict_cxx${std} OBJECT tests/host/second_tu.cpp)
    target_link_libraries(tinyyolo_strict_cxx${std} PRIVATE TinyYolo)
    target_compile_options(tinyyolo_strict_cxx${std} PRIVATE
      ${TINYYOLO_WARNINGS} -fno-exceptions -fno-rtti -Os)
    set_target_properties(tinyyolo_strict_cxx${std} PROPERTIES
      CXX_STANDARD ${std} CXX_STANDARD_REQUIRED ON)
  endforeach()
  add_custom_target(tinyyolo_strict_check DEPENDS tinyyolo_strict_cxx11 tinyyolo_strict_cxx17)
  add_test(NAME tinyyolo_strict_compile
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target tinyyolo_strict_check)
endif()

