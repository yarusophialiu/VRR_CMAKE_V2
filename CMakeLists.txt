enable_language(CUDA)

add_falcor_executable(EncodeDecode)

find_package(CUDAToolkit REQUIRED)

add_dependencies(EncodeDecode GBuffer)
# add_dependencies(EncodeDecode DeferredShading)


target_sources(EncodeDecode PRIVATE
    EncodeDecode.cpp
    EncodeDecode.h
    stb_image_write.h
    ColorSpace.h
    # FramePresenterD3D11.h
    # FramePresenterD3D.h
    EncodeDecode.3d.slang
    EncodeDecode.rt.slang
)

set(NVENC_SDK_ROOT "C:/Video_Codec_SDK_12.1.14/Video_Codec_SDK_12.1.14")
set(NVENC_SDK_LIB_DIR "${NVENC_SDK_ROOT}/Lib/x64")
# set(NVENC_CURL_DIR "C:/curl-8.10.1_2")
# set(NVENC_CURL_LIB_DIR "${NVENC_CURL_DIR}/lib")
set(CODEC_ROOT "C:/cppcodec")

# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

add_library(ColorSpace STATIC ColorSpace.cu)

target_include_directories(EncodeDecode PRIVATE
${NVENC_SDK_ROOT}/Interface ${NVENC_SDK_ROOT}/Samples/Utils
# ${OPENEXR_INCLUDE_DIRS}
${NVENC_CURL_DIR}/include
${CODEC_ROOT}
)

target_link_libraries(EncodeDecode PRIVATE
${NVENC_SDK_LIB_DIR}/nvcuvid.lib
${NVENC_SDK_LIB_DIR}/nvencodeapi.lib
# ${NVENC_CURL_LIB_DIR}/libcurl.dll.a
# ${NVENC_CURL_DIR}/bin/libcurl-x64.dll
CUDA::cudart
ColorSpace
d3d11.lib
dxgi.lib)

# target_link_libraries(EncodeDecode PRIVATE
#     ${NVENC_SDK_LIB_DIR}/nvcuvid.lib
#     ${NVENC_SDK_LIB_DIR}/nvencodeapi.lib
#     CUDA::cudart
#     ColorSpace
#     d3d11.lib
#     dxgi.lib
# )

target_copy_shaders(EncodeDecode Samples/EncodeDecode)

target_source_group(EncodeDecode "Samples")
