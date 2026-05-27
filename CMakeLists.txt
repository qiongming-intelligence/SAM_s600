cmake_minimum_required(VERSION 3.22)

project(sam_s600 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SAM_S600_BUILD_APPS "Build SAM_s600 command line applications" ON)
option(SAM_S600_BUILD_TESTS "Build SAM_s600 tests" OFF)
option(SAM_S600_ENABLE_HOBOT "Enable D-Robotics/Hobot integration hooks" ON)

include(cmake/CompilerOptions.cmake)

add_library(sam_s600
    src/core/profiler.cpp
    src/bpu/bpu_context.cpp
    src/bpu/bpu_model.cpp
    src/bpu/bpu_allocator.cpp
    src/sam3/sam3_model.cpp
    src/sam3/sam3_processor.cpp
    src/sam3/sam3_tokenizer.cpp
    src/sam3/sam3_manifest.cpp
    src/sam3/sam3_image_predictor.cpp
    src/sam3/sam3_video_predictor.cpp
    src/multiplex/multiplex_video_predictor.cpp
    src/postprocess/mask_ops.cpp
)

add_library(sam_s600::sam_s600 ALIAS sam_s600)

target_include_directories(sam_s600
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(sam_s600 PUBLIC cxx_std_17)
target_apply_sam_s600_warnings(sam_s600)

if(SAM_S600_ENABLE_HOBOT AND EXISTS "/usr/hobot/include")
    target_include_directories(sam_s600 PRIVATE /usr/hobot/include)
    target_compile_definitions(sam_s600 PRIVATE SAM_S600_HAS_HOBOT=1)
endif()

if(SAM_S600_ENABLE_HOBOT AND EXISTS "/usr/include/hobot/dnn/hb_dnn.h" AND EXISTS "/usr/hobot/lib/libdnn.so")
    target_compile_definitions(sam_s600 PRIVATE SAM_S600_HAS_HOBOT_DNN=1)
    target_link_libraries(sam_s600 PUBLIC /usr/hobot/lib/libdnn.so)
    if(EXISTS "/usr/hobot/lib/libhbucp.so")
        target_link_libraries(sam_s600 PUBLIC /usr/hobot/lib/libhbucp.so)
    endif()
endif()

if(SAM_S600_ENABLE_HOBOT AND EXISTS "/usr/hobot/include/hb_bpu_mem.h" AND EXISTS "/usr/hobot/lib/libbpu.so")
    target_compile_definitions(sam_s600 PRIVATE SAM_S600_HAS_HOBOT_BPU_MEM=1)
    target_link_libraries(sam_s600 PUBLIC /usr/hobot/lib/libbpu.so)
endif()

if(SAM_S600_BUILD_APPS)
    foreach(app IN ITEMS sam3_image sam3_image_interactive sam3_video sam3_video_tracking sam3_multiplex_video sam3_camera sam3_rtsp sam3_benchmark)
        add_executable(${app} apps/${app}.cpp apps/cli_common.cpp)
        target_link_libraries(${app} PRIVATE sam_s600::sam_s600)
        target_apply_sam_s600_warnings(${app})
    endforeach()
endif()

if(SAM_S600_BUILD_TESTS)
    enable_testing()

    add_executable(bpu_allocator_smoke tests/bpu_allocator_smoke.cpp)
    target_link_libraries(bpu_allocator_smoke PRIVATE sam_s600::sam_s600)
    target_apply_sam_s600_warnings(bpu_allocator_smoke)
    add_test(NAME bpu_allocator_smoke COMMAND bpu_allocator_smoke)

    add_executable(sam3_image_predictor_smoke tests/sam3_image_predictor_smoke.cpp)
    target_link_libraries(sam3_image_predictor_smoke PRIVATE sam_s600::sam_s600)
    target_apply_sam_s600_warnings(sam3_image_predictor_smoke)
    add_test(NAME sam3_image_predictor_smoke COMMAND sam3_image_predictor_smoke)

    add_executable(sam3_video_predictor_smoke tests/sam3_video_predictor_smoke.cpp)
    target_link_libraries(sam3_video_predictor_smoke PRIVATE sam_s600::sam_s600)
    target_apply_sam_s600_warnings(sam3_video_predictor_smoke)
    add_test(NAME sam3_video_predictor_smoke COMMAND sam3_video_predictor_smoke)
endif()
