enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11) # Example for CUDA 11
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_falcor_executable(EncodeDecode)
find_package(CUDAToolkit REQUIRED)

# add_dependencies(EncodeDecode GBuffer DeferredShading)
# add_dependencies(EncodeDecode DeferredShading)

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS}) # include directories from opencv

target_sources(EncodeDecode PRIVATE
    EncodeDecode.cpp
    EncodeDecode.h
    stb_image_write.h
    ColorSpace.h
    # FramePresenterD3D11.h
    # FramePresenterD3D.h
    EncodeDecode.3d.slang
    EncodeDecode.rt.slang
    ComputeVelocity.cs.slang
)

set(NVENC_SDK_ROOT "C:/Video_Codec_SDK_12.1.14/Video_Codec_SDK_12.1.14")
set(NVENC_SDK_LIB_DIR "${NVENC_SDK_ROOT}/Lib/x64")
# set(NVENC_CURL_DIR "C:/curl-8.10.1") # curl-8.10.1 curl-8.10.1_2
# set(NVENC_CURL_LIB_DIR "${NVENC_CURL_DIR}/lib")
set(CODEC_ROOT "C:/cppcodec")
set(ONNXRUNTIME "C:/Users/15142/.nuget/packages/microsoft.ml.onnxruntime.directml/1.20.0/build/native")
set(ONNXRUNTIME_NATIVE "C:/Users/15142/.nuget/packages/microsoft.ml.onnxruntime.directml/1.20.0/runtimes/win-x64/native") # C:\Users\15142\packages\Microsoft.ML.OnnxRuntime.DirectML.1.20.0
add_library(ColorSpace STATIC ColorSpace.cu)

# specify the directories where the compiler should look for header files when building a target
target_include_directories(EncodeDecode PRIVATE
    ${NVENC_SDK_ROOT}/Interface
    ${NVENC_SDK_ROOT}/Samples/Utils
    # ${NVENC_CURL_DIR}/include
    ${CODEC_ROOT}
    ${ONNXRUNTIME}/include
)

link_directories("C:/Users/15142/.nuget/packages/microsoft.ml.onnxruntime.directml/1.20.0/build/native")

target_link_libraries(EncodeDecode PRIVATE
    ${NVENC_SDK_LIB_DIR}/nvcuvid.lib
    ${NVENC_SDK_LIB_DIR}/nvencodeapi.lib
    # ${NVENC_CURL_LIB_DIR}/libcurl.dll.a
    ${ONNXRUNTIME_NATIVE}/onnxruntime.lib
    # ${ONNXRUNTIME_NATIVE}/onnxruntime.dll # DLL files should not be linked directly; they are dynamic libraries that are loaded at runtime, instead link lib
    CUDA::cudart
    ColorSpace
    d3d11.lib
    dxgi.lib
)

target_copy_shaders(EncodeDecode Samples/EncodeDecode)
target_source_group(EncodeDecode "Samples")


# TODO: if rebuild the whole project, manually paste onnxruntime.dll and curl related dll into the same folder as the executable
# onnxruntime.dll: placing the Dynamic Link Library (DLL) file in the same folder as the executable
# an application can load and execute dll at runtime
# .lib are compiled directly into the application, DLLs are separate files that the operating system loads only when needed

# VRRML run_onnx.py runs onnx model
