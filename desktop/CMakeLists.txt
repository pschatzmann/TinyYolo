# Desktop tools (C++, host only): the Darknet blob and weight tools, the YOLO checker and detector
# for each model, and the tinyyolo tool. The library and the tests are in the root CMakeLists.txt.
# The YOLO models are written by tools/yolo/yolo_export.py (the single-file .tym models).

# Host tool: converts a Darknet model into an int8 model blob for flash (see ModelBlob.h).
# AVX2 for the int8 dot products of DotProductBackend (x86-64 only; the scalar code is used otherwise).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  if(MSVC)
    add_compile_options(/arch:AVX2)
  else()
    add_compile_options(-mavx2)
  endif()
endif()

add_executable(tinyyolo_export_blob export_model_blob.cpp)
target_link_libraries(tinyyolo_export_blob PRIVATE TinyYolo)
set_target_properties(tinyyolo_export_blob PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Host tool: writes a Darknet .weights file matching a .cfg (sample models for examples).
add_executable(tinyyolo_make_weights make_darknet_weights.cpp)
target_link_libraries(tinyyolo_make_weights PRIVATE TinyYolo)
set_target_properties(tinyyolo_make_weights PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

# The desktop tool tinyyolo: TinyCamera captures, TinyGPU shows the picture in an SDL2 window.
# The TinyCamera and TinyGPU libraries are built as subprojects from their own folders
# (sibling directories by default). Needs SDL2 (pkg-config name sdl2) for the window.
option(TINYYOLO_BUILD_CAMERA_VIEW "Build the desktop tool tinyyolo (webcam, pictures, board stream; needs SDL2)" OFF)
if(TINYYOLO_BUILD_CAMERA_VIEW)
  set(TINYCAMERA_DIR "${CMAKE_SOURCE_DIR}/../TinyCamera" CACHE PATH "TinyCamera library directory")
  set(TINYGPU_DIR "${CMAKE_SOURCE_DIR}/../TinyGPU" CACHE PATH "TinyGPU library directory")
  set(TINYMD_DIR "${CMAKE_SOURCE_DIR}/../TinyMaterialDesign" CACHE PATH "TinyMaterialDesign library directory")
  set(TINYJPEG_DIR "${CMAKE_SOURCE_DIR}/../TinyJPEG" CACHE PATH "TinyJPEG library directory (header-only JPEG decoder)")
  find_package(PkgConfig REQUIRED)
  find_package(Threads REQUIRED)
  pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)

  # The tool needs only the capture (V4L2 on Linux, no JPEG) and the display.
  set(TINYCAMERA_USE_TINYJPEG OFF CACHE BOOL "" FORCE)
  add_subdirectory(${TINYCAMERA_DIR} ${CMAKE_BINARY_DIR}/TinyCamera EXCLUDE_FROM_ALL)
  add_subdirectory(${TINYGPU_DIR} ${CMAKE_BINARY_DIR}/TinyGPU EXCLUDE_FROM_ALL)

  # tinyyolo: a Darknet model or a YOLO export, on the webcam, on still pictures or on a board stream.
  add_executable(tinyyolo tinyyolo/tinyyolo.cpp)
  # TinyMaterialDesign is header-only: its src folder is added, and its TinyGPU dependency is
  # the TinyGPU::TinyGPU target above.
  target_include_directories(tinyyolo PRIVATE ${TINYMD_DIR}/src ${TINYJPEG_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/../tools/yolo)
  target_link_libraries(tinyyolo PRIVATE TinyYolo TinyCamera::TinyCamera TinyGPU::TinyGPU PkgConfig::SDL2 Threads::Threads)
  set_target_properties(tinyyolo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  # The convolution kernels vectorize best for the machine they run on.
  option(TINYYOLO_NATIVE_ARCH "Optimise the desktop tool for this machine (-march=native)" ON)
  if(TINYYOLO_NATIVE_ARCH)
    target_compile_options(tinyyolo PRIVATE -march=native)
  endif()
  # The default model: the YOLO26n model file of this repository (the fastest on the desktop).
  target_compile_definitions(tinyyolo PRIVATE "TINYYOLO_DEFAULT_MODEL=\"${CMAKE_CURRENT_SOURCE_DIR}/../models/yolo26n/yolo26n.tym\"")
  # The model of the board stream (the CameraStream example runs YOLO11n); its names label the boxes.
  target_compile_definitions(tinyyolo PRIVATE "TINYYOLO_BOARD_MODEL=\"${CMAKE_CURRENT_SOURCE_DIR}/../models/yolo11n/yolo11n.tym\"")
endif()

