include(${CMAKE_CURRENT_LIST_DIR}/cmake/CompilerOptions.cmake)

add_library(sam_s600
    core/profiler.cpp
    bpu/bpu_context.cpp
    bpu/bpu_model.cpp
    bpu/bpu_allocator.cpp
    sam3/sam3_model.cpp
    sam3/sam3_processor.cpp
    sam3/sam3_tokenizer.cpp
    sam3/sam3_manifest.cpp
    sam3/sam3_image_predictor.cpp
    sam3/sam3_video_predictor.cpp
    multiplex/multiplex_video_predictor.cpp
    postprocess/mask_ops.cpp
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

add_library(sam_s600_cli
    cli/manifest_cli.cpp
)
add_library(sam_s600::cli ALIAS sam_s600_cli)
target_link_libraries(sam_s600_cli PUBLIC sam_s600::sam_s600)
target_apply_sam_s600_warnings(sam_s600_cli)

if(SAM_S600_BUILD_APPS)
    foreach(app IN ITEMS sam3_image sam3_image_interactive sam3_video sam3_video_tracking sam3_multiplex_video sam3_camera sam3_rtsp sam3_benchmark)
        add_executable(${app} bin/${app}.cpp)
        target_link_libraries(${app} PRIVATE sam_s600::cli)
        target_apply_sam_s600_warnings(${app})
    endforeach()
endif()
