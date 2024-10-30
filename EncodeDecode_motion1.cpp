// EncodeDecode.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once
#define TINYEXR_IMPLEMENTATION

#include <iostream>
#include <d3d12.h>
#include <d3d11.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <iomanip>
#include <vector>
#include <fstream>

#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/TextRenderer.h"
#include "Core/API/NativeHandleTraits.h"
#include "RenderGraph/RenderGraph.h"

// #include <OpenEXR/ImfRgbaFile.h>
// #include <OpenEXR/ImfArray.h>

#include "ColorSpace.h"
#include "NvCodecUtils.h"
#include "EncodeDecode.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <Windows.h>
#include <wingdi.h>
#include "FramePresenterD3D11.h"

#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cstdio>

namespace fs = std::filesystem;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

FALCOR_EXPORT_D3D12_AGILITY_SDK
simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

// Macor for making NVENC API calls and reporting errors when they fail
#define NVENC_API_CALL(nvencAPI)                                               \
    do                                                                         \
    {                                                                          \
        NVENCSTATUS errorCode = nvencAPI;                                      \
        if (errorCode != NV_ENC_SUCCESS)                                       \
        {                                                                      \
            std::cout << #nvencAPI << " returned error " << errorCode << "\n"; \
        }                                                                      \
    } while (0)

#define NVDEC_API_CALL(cuvidAPI)                                               \
    do                                                                         \
    {                                                                          \
        CUresult errorCode = cuvidAPI;                                         \
        if (errorCode != CUDA_SUCCESS)                                         \
        {                                                                      \
            std::cout << #cuvidAPI << " returned error " << errorCode << "\n"; \
        }                                                                      \
    } while (0)

#define CUDA_DRVAPI_CALL(call)                                          \
    do                                                                  \
    {                                                                   \
        CUresult err__ = call;                                          \
        if (err__ != CUDA_SUCCESS)                                      \
        {                                                               \
            const char* szErrName = NULL;                               \
            cuGetErrorName(err__, &szErrName);                          \
            std::cout << "CUDA driver API error " << szErrName << "\n"; \
        }                                                               \
    } while (0)

// Util function mapping ecoder pixel format ot DX12 pixel format
DXGI_FORMAT GetD3D12Format(NV_ENC_BUFFER_FORMAT eBufferFormat)
{
    switch (eBufferFormat)
    {
    case NV_ENC_BUFFER_FORMAT_ARGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

#include <fstream>
#include <cstdint>

#pragma pack(push, 1)
struct BMPHeader
{
    uint16_t signature;       // "BM"
    uint32_t fileSize;        // Size of the BMP file in bytes
    uint32_t reserved;        // Reserved, set to 0
    uint32_t dataOffset;      // Offset of pixel data from the beginning of the BMP file
    uint32_t headerSize;      // Size of this header (40 bytes)
    int32_t width;            // Width of the image in pixels
    int32_t height;           // Height of the image in pixels
    uint16_t planes;          // Number of color planes, must be 1
    uint16_t bitsPerPixel;    // Number of bits per pixel (e.g., 32 for RGBA)
    uint32_t compression;     // Compression method being used (0 for uncompressed)
    uint32_t dataSize;        // Size of the raw bitmap data (including padding)
    int32_t horizontalRes;    // Pixels per meter
    int32_t verticalRes;      // Pixels per meter
    uint32_t numColors;       // Number of colors in the color palette (0 for 32-bit)
    uint32_t importantColors; // Number of important colors used (0 for 32-bit)
};
#pragma pack(pop)


void saveVectorToFile(const std::vector<uint8_t>& val, const std::string& filename) {
    // const std::vector<uint8_t>  buffer(val);
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        throw std::ios_base::failure("Failed to open file for writing");
    }

    // size_t size = buffer.size();
    // file.write((char*)&size, sizeof(size));

    // file.write(reinterpret_cast<const char*>(buffer.data()), val.size()); //  (float*)val.data()
    file.write(reinterpret_cast<const char*>(val.data()), val.size());
    if (!file) {
        throw std::ios_base::failure("Failed to write data to file");
    }
    file.close();
}



void writeBMP(const char* filename, uint8_t* imageData, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }

    // BMP header
    BMPHeader bmpHeader = {0};
    bmpHeader.signature = 0x4D42;                                // "BM"
    bmpHeader.fileSize = sizeof(BMPHeader) + width * height * 4; // 4 channels (RGBA)
    bmpHeader.dataOffset = sizeof(BMPHeader);
    bmpHeader.headerSize = sizeof(BMPHeader) - 14;
    bmpHeader.width = width;
    bmpHeader.height = -height;
    bmpHeader.planes = 1;
    bmpHeader.bitsPerPixel = 32;
    bmpHeader.compression = 0;
    bmpHeader.dataSize = width * height * 4;
    bmpHeader.horizontalRes = 2835; // Pixels per meter (72 DPI)
    bmpHeader.verticalRes = 2835;   // Pixels per meter (72 DPI)

    // Write the BMP header
    file.write(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader));

    // Write the pixel data (assuming RGBA format)
    file.write(reinterpret_cast<char*>(imageData), width * height * 4);

    // file.close();
}

// static const Falcor::float4 kClearColor(0.38f, 0.52f, 0.10f, 1);
static const Falcor::float4 kClearColor(0.5f, 0.16f, 0.098f, 1);
/*
Arcade/Arcade.pyscene
test_scenes/two_volumes.pyscene
test_scenes/grey_and_white_room/grey_and_white_room.fbx
SunTemple_v4/SunTemple_v4/SunTemple/SunTemple.pyscene
Bistro/Bistro/BistroInterior_Wine.fbx
Bistro/Bistro/BistroInterior.fbx
Bistro/Bistro/BistroExterior.fbx
*/
// static const std::string kDefaultScene = "SunTemple_v4/SunTemple_v4/SunTemple/SunTemple.pyscene";
// static const std::string kDefaultScene = "Arcade/Arcade.pyscene";

// constructor
EncodeDecode::EncodeDecode(const SampleAppConfig& config) : SampleApp(config)
{
    /*
    1422x800 dec not working
    new pairs:
    1536, 1200
    864, 676
    */
    mWidth = config.windowDesc.width;   // 1920, 4096, 1280, 854, 640, 960, 1024, 1280, 1440, 1423
    mHeight = config.windowDesc.height; // 1080, 2160, 720, 480, 360, 540, 600, 800, 900, 800
    // int nEncoders = NvEncGetEncodeProfileGUIDCount();


    std::cout << '\n';
    std::cout << "mWidth: " << mWidth << std::endl;
    std::cout << "mHeight: " << mHeight << std::endl;
    std::cout << '\n';

    // fpOut is an instance of std::ofstream that is associated
    // with the specified output file (szOutFilePath) and is opened in binary output mode
    // std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
    if (outputEncodedFrames)
    {
        // generate outputfile name with timestamp
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S"); // Format the time as a string
        std::string timestamp = ss.str();
        std::string newFilePath = "encodedOutput264/out_" + timestamp + ".h264";
        strncpy(szOutFilePath, newFilePath.c_str(), sizeof(szOutFilePath));
        szOutFilePath[sizeof(szOutFilePath) - 1] = '\0'; // Ensure null-termination
        std::cout << "szOutFilePath: " << szOutFilePath << std::endl;

        fpEncOut = new std::ofstream(szOutFilePath, std::ios::out | std::ios::binary);

        if (!fpEncOut)
        {
            std::ostringstream err;
            err << "Unable to open output file: " << szOutFilePath << std::endl;
            throw std::invalid_argument(err.str());
        }
    }

    // const ref<Device>& device = getDevice(); // get from falcor
    // mpDevice = device.get();
    mpDevice = getDevice();

    mpD3D12Device = mpDevice->getNativeHandle().as<ID3D12Device*>();

    mpDecodeFence = mpDevice->createFence();
    mNInputFenceVal = 0;
    mNOutputFenceVal = 0;
    mNDecodeFenceVal = 0;


    mNVEnc = {};
    mEBufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
    mCudaDevice = 0;

    // frame buffer we will render into, mpRtOut is falcor's object
    // the frame size here determines the windows size of reference frames

    mpRtOut = getDevice()->createTexture2D(
        //mWidth, mHeight, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        // TODO: change the width and height of the reference frame size // 1920, 1080, 854, 480
        mWidth, mHeight, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );


    mipLevelsCompute = fmax(ceil(log2(mWidth)), ceil(log2(mHeight)));
    // mipLevels = fmax(ceil(log2(mWidth/64)), ceil(log2(mHeight/64)));
    mipLevels = 1; // 7: 64x64 patches, 11: 1x1 patch, 1: 1920x1080 patch
    std::cout << "constructor mipLevels: " << mipLevels << "\n";

    mpRtMip = getDevice()->createTexture2D(
        // mWidth, mHeight, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess |
        // ResourceBindFlags::ShaderResource
        //  TODO: change the width and height of the reference frame size // 1920, 1080, 854, 480
        mWidth, mHeight, ResourceFormat::RG32Float, 1, mipLevels, nullptr,
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget
    );


    //// cast into directx 12
    //// falcor's device, createtexture3d
    // mPDecoderOutputTexture = getDevice()->createTexture2D(
    //     mWidth, mHeight, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess |
    //     ResourceBindFlags::ShaderResource
    //)->getNativeHandle().as<ID3D12Resource*>();
}

EncodeDecode::~EncodeDecode() {}

/// what happens when scene gets loaded
// parent class sampleapp calls onload when start running
void EncodeDecode::onLoad(RenderContext* pRenderContext)
{
    if (getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false)
    {
        FALCOR_THROW("Device does not support raytracing!");
    }

    initEncoder();
    // initDecoder();
    std::cout << "load scene: " << std::endl;

    loadScene(kDefaultScene, getTargetFbo().get());

    Properties gBufferProps = {};


    mpRenderGraph = RenderGraph::create(mpDevice, "EncodeDecode");

    mpRenderGraph->createPass("GBuffer", "GBufferRaster", gBufferProps);
    mpRenderGraph->onResize(getTargetFbo().get());
    mpRenderGraph->setScene(mpScene);
    //mpRenderGraph->addEdge("GBuffer.mvec");
    mpRenderGraph->markOutput("GBuffer.mvec");
    //mpRenderGraph->markOutput("GBuffer.diffuseOpacity");

    //// std::cout << "mpCamera: (" << currPos.x << ", " << currPos.y << ", " << currPos.z << ")\n";
}

/*
TODO:change h, w
resize window changes the size of presenter of the decoded frame
*/
void EncodeDecode::onResize(uint32_t width, uint32_t height)
{
    float h = (float)height; // 1080, 2160, 720,
    float w = (float)width;  // 1920, 3840, 1280,

    std::cout << '\n';
    std::cout << "Width: " << w << std::endl;
    std::cout << "Height: " << h << std::endl;
    std::cout << '\n';

    if (mpCamera)
    {
        mpCamera->setFocalLength(18);
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }
}


void EncodeDecode::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

//     // std::cout << "mpCamera: (" << currPos.x << ", " << currPos.y << ", " << currPos.z << ")\n";
    // mpCamera->setPosition(currPos);
    mpRenderGraph->execute(pRenderContext);
    //motionVectorResource = mpRenderGraph->getOutput("GBuffer.diffuseOpacity");
    motionVectorResource = mpRenderGraph->getOutput("GBuffer.mvec");
    motionVectorTexture = static_ref_cast<Texture>(motionVectorResource);


    //pRenderContext->blit(motionVectorResource->getSRV(), mpRtMip->getRTV());

    /* std::cout << "mpCamera: (" << mpCamera->getPosition().x << ", " << mpCamera->getPosition().y << ", " << mpCamera->getPosition().z
               << ")\n";
     std::cout << "\n";*/

    // Ugly hack, just to get consistent videos
    static double timeSecs = 0;

    if (mpScene)
    {
        Scene::UpdateFlags updates = mpScene->update(pRenderContext, speed * timeSecs); // fast 2, normal 1, slow 0.5
        if (is_set(updates, Scene::UpdateFlags::GeometryChanged))
            FALCOR_THROW("This sample does not support scene geometry changes.");
        if (is_set(updates, Scene::UpdateFlags::RecompileNeeded))
            FALCOR_THROW("This sample does not support scene changes that require shader recompilation.");

        static uint32_t fcount = 0;
        static int fCount_rt = 0;

        InterlockedIncrement(&mNDecodeFenceVal);

        if (mRayTrace)
            renderRT(pRenderContext, pTargetFbo, fcount);
        else
            renderRaster(pRenderContext, pTargetFbo);

        cpuWaitForFencePoint(mpDecodeFence->getNativeHandle().as<ID3D12Fence*>(), mNDecodeFenceVal);

        // // if (outputReferenceFrames && (fCount_rt >= frameRate))
        // if (fCount_rt >= frameRate)
        // {
        //     // std::cout << "count fcount " << fcount << "\n";
        //     std::cout << "count fCount_rt " << fCount_rt << "\n";

        // }

        // std::cout << "============================================== end ==============================================" << "\n";


        // Sleep(100); // miliseconds,  avoid tearing
        encodeFrameBuffer();
        // decodeFrameBuffer();

        //   write to bmp file
        if (outputDecodedFrames && outputReferenceFrames)
        {
            snprintf(szDecOutFilePath, sizeof(szDecOutFilePath), "%s%d.bmp", decBaseFilePath, fcount);
            writeBMP(szDecOutFilePath, mPHostRGBAFrame, mWidth, mHeight);

            snprintf(szRefOutFilePath, sizeof(szRefOutFilePath), "%s%d.bmp", refBaseFilePath, fcount);
            mpRtOut->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);
        }
        else if (outputDecodedFrames)
        {
            // Use snprintf to format the string with the count
            snprintf(szDecOutFilePath, sizeof(szDecOutFilePath), "%s%d.bmp", decBaseFilePath, fcount);
            writeBMP(szDecOutFilePath, mPHostRGBAFrame, mWidth, mHeight);
        }

        if (frameLimit > 0 && fcount >= frameLimit)
        {
            std::exit(0);
        }

        fCount_rt += 1;
        ++fcount;
        timeSecs += 1.0 / frameRate;
    }

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
}

void EncodeDecode::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "VRR Settings", {300, 400}, {10, 80});

    w.checkbox("Ray Trace", mRayTrace);
    w.checkbox("Use Depth of Field", mUseDOF);
    if (w.button("Load Scene"))
    {
        std::filesystem::path path;
        if (openFileDialog(Scene::getFileExtensionFilters(), path))
        {
            loadScene(path, getTargetFbo().get());
        }
    }

    mpScene->renderUI(w);
}

bool EncodeDecode::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.key == Input::Key::Space && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        mRayTrace = !mRayTrace;
        return true;
    }

    if (mpScene && mpScene->onKeyEvent(keyEvent))
        return true;

    return false;
}

bool EncodeDecode::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene && mpScene->onMouseEvent(mouseEvent);
}

/*
set default params, e.g. framerate
rateControlMode = NV_ENC_PARAMS_RC_CONSTQP, constant quality
*/
void EncodeDecode::makeDefaultEncodingParams(
    NV_ENC_INITIALIZE_PARAMS* pIntializeParams,
    GUID codecGuid,
    GUID presetGuid,
    NV_ENC_TUNING_INFO tuningInfo
)
{
    // Zero the param memory, all params set to 0 use the defualt values
    memset(pIntializeParams->encodeConfig, 0, sizeof(NV_ENC_CONFIG));
    auto pEncodeConfig = pIntializeParams->encodeConfig;
    memset(pIntializeParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
    pIntializeParams->encodeConfig = pEncodeConfig;
    pIntializeParams->encodeConfig->version = NV_ENC_CONFIG_VER;
    pIntializeParams->version = NV_ENC_INITIALIZE_PARAMS_VER;

    // Set the basic encoding params, most of what we will be changing is set here.
    pIntializeParams->encodeGUID = codecGuid;
    pIntializeParams->presetGUID = presetGuid;
    pIntializeParams->encodeWidth = mWidth;
    pIntializeParams->encodeHeight = mHeight;
    pIntializeParams->darWidth = mWidth;
    pIntializeParams->darHeight = mHeight;
    // Bitrate = Frame Rate × Frame Size × Bits per Pixel
    pIntializeParams->frameRateNum = frameRate; // numerator
    pIntializeParams->frameRateDen = 1;  // denominator, num/den = framerate
    pIntializeParams->enablePTD = 1;
    pIntializeParams->reportSliceOffsets = 0;
    pIntializeParams->enableSubFrameWrite = 0;
    pIntializeParams->maxEncodeWidth = mWidth;
    pIntializeParams->maxEncodeHeight = mHeight;
    pIntializeParams->enableOutputInVidmem = 0;
    pIntializeParams->enableEncodeAsync = 0; // TODO: async 0: async = 0 is sync mode, 1 is async mode

    // Use the presets to set other params, we may want to change this later
    NV_ENC_PRESET_CONFIG presetConfig = {NV_ENC_PRESET_CONFIG_VER, {NV_ENC_CONFIG_VER}};
    NVENC_API_CALL(mNVEnc.nvEncGetEncodePresetConfig(mHEncoder, codecGuid, presetGuid, &presetConfig));
    memcpy(pIntializeParams->encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
    pIntializeParams->encodeConfig->frameIntervalP = 1;
    pIntializeParams->encodeConfig->gopLength = NVENC_INFINITE_GOPLENGTH;

    /*
    framerate is only used in rate control, how aggressively to encode a frame
    constant quantization param: quantization parameter (QP) is kept constant throughout the
    entire video sequence,  compression level remains consistent for the entire video.
    The quantization parameter influences the trade-off between video quality and file size.
    */
    pIntializeParams->encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR; // NV_ENC_PARAMS_RC_CONSTQP, NV_ENC_PARAMS_RC_CBR,
                                                                                     // NV_ENC_PARAMS_RC_VBR
    // mEncodeconfig.rcParams.constQP = {28, 31, 25};
    // pIntializeParams->encodeConfig->rcParams.constQP = {28, 31, 25}; // TODO: why set it like // ignored in CBR

    if (true)
    {
        pIntializeParams->tuningInfo = tuningInfo;
        presetConfig = {NV_ENC_PRESET_CONFIG_VER, {NV_ENC_CONFIG_VER}};
        mNVEnc.nvEncGetEncodePresetConfigEx(mHEncoder, codecGuid, presetGuid, tuningInfo, &presetConfig);
        memcpy(pIntializeParams->encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
    }

    // Set per codec options
    if (pIntializeParams->encodeGUID == NV_ENC_CODEC_H264_GUID)
    {
        if (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
        {
            pIntializeParams->encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 3;
        }
        pIntializeParams->encodeConfig->encodeCodecConfig.h264Config.idrPeriod = pIntializeParams->encodeConfig->gopLength;
    }
    else if (pIntializeParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID)
    {
        pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 =
            (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 2 : 0;
        if (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
        {
            pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
        }
        pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = pIntializeParams->encodeConfig->gopLength;
    }
    else if (pIntializeParams->encodeGUID == NV_ENC_CODEC_AV1_GUID)
    {
        pIntializeParams->encodeConfig->encodeCodecConfig.av1Config.pixelBitDepthMinus8 =
            (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT) ? 2 : 0;
        pIntializeParams->encodeConfig->encodeCodecConfig.av1Config.inputPixelBitDepthMinus8 =
            (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT) ? 2 : 0;
        pIntializeParams->encodeConfig->encodeCodecConfig.av1Config.chromaFormatIDC = 1;
        pIntializeParams->encodeConfig->encodeCodecConfig.av1Config.idrPeriod = pIntializeParams->encodeConfig->gopLength;
        pIntializeParams->encodeConfig->frameIntervalP = 1;
    }

    pIntializeParams->bufferFormat = mEBufferFormat;
}

void EncodeDecode::initEncoder()
{
    // Open the NVEnc API (function pointer table stored in mNVEnc)
    mNVEnc = {NV_ENCODE_API_FUNCTION_LIST_VER};

    NVENCSTATUS errorCode = NvEncodeAPICreateInstance(&mNVEnc);

    if (errorCode != NV_ENC_SUCCESS)
    {
        printf("Error opening NVENC api");
    }

    // Open encoder session, get parameters
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = {NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
    encodeSessionExParams.device = mpD3D12Device;
    encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_DIRECTX;
    encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
    NVENC_API_CALL(mNVEnc.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &mHEncoder));

    // Get encoder params, here we use the defualt ultra-low latency, low quality presets for some of the params
    // Encode Session Initialization parameters
    mEncoderInitializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    // Encoder configuration parameters to be set during initialization
    mEncodeConfig = {NV_ENC_CONFIG_VER};
    mEncoderInitializeParams.encodeConfig = &mEncodeConfig;

    /*
    set encoding parameter, use h264 here
    NV_ENC_CODEC_H264_GUID is a constant representing the unique identifier (GUID) for the H.264 video codec
    NV_ENC_CONFIG_HEVC = h265, NV_ENC_CONFIG_AV1 = av1, NV_ENC_CODEC_HEVC_GUID
    NVENC API is part of the NVIDIA Video Codec SDK, providing access to encoding capabilities of NVIDIA GPUs
    P1 preset achieves high encoding speed but might sacrifice video quality and compression efficiency
    params, codecguid, presetguid, tunning info
    GUID: global unique identifier, think of it as a constant
    */

    makeDefaultEncodingParams(
        &mEncoderInitializeParams, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY
    ); // TODO: changed NV_ENC_PRESET_P1_GUID

    if (mpD3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mpInputFence)) != S_OK)
    {
        // Error
    }

    if (mpD3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mpOutputFence)) != S_OK)
    {
        // Error
    }

    if (((uint32_t)mEncodeConfig.frameIntervalP) > mEncodeConfig.gopLength)
    {
        mEncodeConfig.frameIntervalP = mEncodeConfig.gopLength;
    }

    mEncodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH; // The number of frames in a GOP (group of pictures)
    mEncodeConfig.frameIntervalP = 1;
    mEncodeConfig.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH; // TODO: h264 only, add h265

    /*
    mEncodeConfig.version = NV_ENC_CONFIG_VER;
    mEncodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    mEncodeConfig.rcParams.constQP = {28, 31, 25};
    mEncoderInitializeParams.encodeConfig = &mEncodeConfig;
    */

    // mEncodeConfig.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;

    // set bitrate 500000 (low quality) 1000000 1200000 (1200 - standard definition)
    // 3000000 4000000 (4000 - hd) 5000000, 8000000 (full hd) 10000000 15 Mbps - 30 Mbps 30000000 (4k)
    mEncodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP; // NV_ENC_PARAMS_RC_CONSTQP NV_ENC_PARAMS_RC_CBR
    mEncodeConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION; // not valid for h264
    mEncodeConfig.rcParams.averageBitRate = bitRate;
    mEncodeConfig.rcParams.vbvBufferSize =
        (mEncodeConfig.rcParams.averageBitRate * mEncoderInitializeParams.frameRateDen / mEncoderInitializeParams.frameRateNum) * 5;
    mEncodeConfig.rcParams.maxBitRate = mEncodeConfig.rcParams.averageBitRate;
    mEncodeConfig.rcParams.vbvInitialDelay = mEncodeConfig.rcParams.vbvBufferSize;

    std::cout << "\naverageBitRate " << mEncodeConfig.rcParams.averageBitRate << "\n";

    NVENC_API_CALL(mNVEnc.nvEncInitializeEncoder(mHEncoder, &mEncoderInitializeParams));

    mNEncoderBuffer = mEncodeConfig.frameIntervalP + mEncodeConfig.rcParams.lookaheadDepth;

    mEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // allocate memory so encoder can work with what we need
    // one buffer each time
    makeEncoderInputBuffers(1);
    makeEncoderOutputBuffers(1);

    mVPCompletionEvent.resize(1, nullptr);

    for (uint32_t i = 0; i < mVPCompletionEvent.size(); i++)
    {
        mVPCompletionEvent[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
    }
}

NV_ENC_REGISTERED_PTR EncodeDecode::registerNVEncResource(
    void* pBuffer,
    NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
    int width,
    int height,
    int pitch,
    NV_ENC_BUFFER_FORMAT bufferFormat,
    NV_ENC_BUFFER_USAGE bufferUsage,
    NV_ENC_FENCE_POINT_D3D12* pInputFencePoint
)
{
    NV_ENC_REGISTER_RESOURCE registerResource = {NV_ENC_REGISTER_RESOURCE_VER};
    registerResource.resourceType = eResourceType;
    registerResource.resourceToRegister = pBuffer;
    registerResource.width = width;
    registerResource.height = height;
    registerResource.pitch = pitch;
    registerResource.bufferFormat = bufferFormat;
    registerResource.bufferUsage = bufferUsage;
    registerResource.pInputFencePoint = pInputFencePoint;
    NVENC_API_CALL(mNVEnc.nvEncRegisterResource(mHEncoder, &registerResource));

    return registerResource.registeredResource;
}

uint32_t EncodeDecode::getEncoderNumChromaPlanes(const NV_ENC_BUFFER_FORMAT bufferFormat)
{
    switch (bufferFormat)
    {
    case NV_ENC_BUFFER_FORMAT_NV12:
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
        return 1;
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
    case NV_ENC_BUFFER_FORMAT_YUV444:
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
        return 2;
    case NV_ENC_BUFFER_FORMAT_ARGB:
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    case NV_ENC_BUFFER_FORMAT_AYUV:
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ABGR10:
        return 0;
    default:
        return -1;
    }
}

int EncodeDecode::getEncoderFrameSize()
{
    switch (mEBufferFormat)
    {
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
    case NV_ENC_BUFFER_FORMAT_NV12:
        return mWidth * (mHeight + (mHeight + 1) / 2);
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
        return 2 * mWidth * (mHeight + (mHeight + 1) / 2);
    case NV_ENC_BUFFER_FORMAT_YUV444:
        return mWidth * mHeight * 3;
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
        return 2 * mWidth * mHeight * 3;
    case NV_ENC_BUFFER_FORMAT_ARGB:
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    case NV_ENC_BUFFER_FORMAT_AYUV:
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ABGR10:
        return 4 * mWidth * mHeight;
    default:
        return 0;
    }
}

uint32_t EncodeDecode::getEncoderOutputBufferSize()
{
    uint32_t bufferSize = getEncoderFrameSize() * 4;
    bufferSize = ALIGN_UP(bufferSize, 4);

    return bufferSize;
}

void EncodeDecode::makeEncoderInputBuffers(int32_t numInputBuffers)
{
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = mWidth;
    resourceDesc.Height = mHeight;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    resourceDesc.Format = GetD3D12Format(mEBufferFormat);

    mVInputBuffers.resize(numInputBuffers);

    for (int i = 0; i < numInputBuffers; i++)
    {
        if (mpD3D12Device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mVInputBuffers[i])
            ) != S_OK)
        {
            // Error
            // NVENC_THROW_ERROR("Failed to create ID3D12Resource", NV_ENC_ERR_OUT_OF_MEMORY);
        }
    }

    registerEncoderInputResources(mWidth, mHeight, mEBufferFormat);

    // Create NV_ENC_INPUT_RESOURCE_D3D12
    for (int i = 0; i < numInputBuffers; i++)
    {
        NV_ENC_INPUT_RESOURCE_D3D12* pInpRsrc = new NV_ENC_INPUT_RESOURCE_D3D12();
        memset(pInpRsrc, 0, sizeof(NV_ENC_INPUT_RESOURCE_D3D12));
        pInpRsrc->inputFencePoint.pFence = mpInputFence;

        mVInputRsrc.push_back(pInpRsrc);
    }
}

void EncodeDecode::makeEncoderOutputBuffers(uint32_t numOutputBuffers)
{
    HRESULT hr = S_OK;

    D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_COPY_DEST;

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = getEncoderOutputBufferSize();
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    mVOutputBuffers.resize(numOutputBuffers);

    for (uint32_t i = 0; i < numOutputBuffers; ++i)
    {
        if (mpD3D12Device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, initialResourceState, nullptr, IID_PPV_ARGS(&mVOutputBuffers[i])
            ) != S_OK)
        {
            // NVENC_THROW_ERROR("Failed to create output ID3D12Resource", NV_ENC_ERR_OUT_OF_MEMORY);
        }
    }

    registerEncoderOutputResources(getEncoderOutputBufferSize());

    for (uint32_t i = 0; i < mVOutputBuffers.size(); ++i)
    {
        NV_ENC_OUTPUT_RESOURCE_D3D12* pOutRsrc = new NV_ENC_OUTPUT_RESOURCE_D3D12();
        memset(pOutRsrc, 0, sizeof(NV_ENC_OUTPUT_RESOURCE_D3D12));
        pOutRsrc->outputFencePoint.pFence = mpOutputFence;
        mVOutputRsrc.push_back(pOutRsrc);
    }
}

void EncodeDecode::registerEncoderInputResources(int width, int height, NV_ENC_BUFFER_FORMAT bufferFormat)
{
    // for (uint32_t i = 0; i < mVInputBuffers.size(); ++i)
    //{
    /*
    NV_ENC_FENCE_POINT_D3D12 is used for specifying a synchronization in the DirectX 12 (D3D12) context.
    This structure is often employed when registering resources with NVENC to ensure proper synchronization between the CPU and GPU.
    */
    NV_ENC_FENCE_POINT_D3D12 regRsrcInputFence;

    // Set input fence point
    memset(&regRsrcInputFence, 0, sizeof(NV_ENC_FENCE_POINT_D3D12));
    regRsrcInputFence.pFence = mpInputFence;
    regRsrcInputFence.waitValue = mNInputFenceVal;
    regRsrcInputFence.bWait = true;

    // atomically increment the value of a variable
    // returns the incremented value
    InterlockedIncrement(&mNInputFenceVal);

    regRsrcInputFence.signalValue = mNInputFenceVal;
    regRsrcInputFence.bSignal = true;

    // get directx 12 resource information
    auto dx12InputTexture = mpRtOut->getNativeHandle().as<ID3D12Resource*>();
    D3D12_RESOURCE_DESC desc = dx12InputTexture->GetDesc();

    // Registering the DirectX 12 Resource with NVENC
    NV_ENC_REGISTERED_PTR registeredPtr = registerNVEncResource(
        dx12InputTexture,                   // DirectX 12 resource to be registered
        NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, // Type of the input resource (DirectX texture in this case)
        width,                              // Width of the resource
        height,                             // Height of the resource
        0,                                  // Pitch (row pitch of the resource), 0 for automatic calculation
        bufferFormat,                       // Format of the input buffer (e.g., DXGI_FORMAT_R8G8B8A8_UNORM)
        NV_ENC_INPUT_IMAGE,                 // Type of input image (e.g., for image input)
        &regRsrcInputFence                  // Pointer to an NV_ENC_FENCE_POINT_D3D12 structure
    );

    // Creating an NvEncInputFrame
    // preparing data structures for use in the NVENC encoding process
    NvEncInputFrame inputframe = {};                            // {}: initialize all its members set to their default value
    ID3D12Resource* pTextureRsrc = dx12InputTexture;            // dx12InputTexture is presumably a pointer to a DirectX 12 resource
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint[2]; // describe the layout of a resource in memory. The [2] suggests that it's
                                                                // an array with two elements.

    mpD3D12Device->GetCopyableFootprints(&desc, 0, 1, 0, inputUploadFootprint, nullptr, nullptr, nullptr);

    inputframe.inputPtr = (void*)dx12InputTexture;
    inputframe.numChromaPlanes = getEncoderNumChromaPlanes(bufferFormat);
    inputframe.bufferFormat = bufferFormat;
    inputframe.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;

    inputframe.pitch = inputUploadFootprint[0].Footprint.RowPitch;

    // Storing Registered Resources and Input Frames
    // a process of accumulating registered resources
    mVRegisteredResources.push_back(registeredPtr);
    mVInputFrames.push_back(inputframe);
    mVMappedInputBuffers.resize(1);

    // CPU wait for register resource to finish
    cpuWaitForFencePoint((ID3D12Fence*)regRsrcInputFence.pFence, regRsrcInputFence.signalValue);
    //}
}

void EncodeDecode::registerEncoderOutputResources(uint32_t bfrSize)
{
    for (uint32_t i = 0; i < mVOutputBuffers.size(); ++i)
    {
        NV_ENC_REGISTERED_PTR registeredPtr = registerNVEncResource(
            mVOutputBuffers[i], NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, bfrSize, 1, 0, NV_ENC_BUFFER_FORMAT_U8, NV_ENC_OUTPUT_BITSTREAM
        );

        mVRegisteredResourcesOutputBuffer.push_back(registeredPtr);
    }

    mVMappedOutputBuffers.resize(mVOutputBuffers.size());
}

void EncodeDecode::mapEncoderResource(uint32_t bufferIndex)
{
    NV_ENC_MAP_INPUT_RESOURCE mapInputResource = {NV_ENC_MAP_INPUT_RESOURCE_VER};
    mapInputResource.registeredResource = mVRegisteredResources[bufferIndex];
    NVENC_API_CALL(mNVEnc.nvEncMapInputResource(mHEncoder, &mapInputResource));
    mVMappedInputBuffers[bufferIndex] = mapInputResource.mappedResource;

    NV_ENC_MAP_INPUT_RESOURCE mapInputResourceBitstreamBuffer = {NV_ENC_MAP_INPUT_RESOURCE_VER};
    mapInputResourceBitstreamBuffer.registeredResource = mVRegisteredResourcesOutputBuffer[bufferIndex];
    NVENC_API_CALL(mNVEnc.nvEncMapInputResource(mHEncoder, &mapInputResourceBitstreamBuffer));
    mVMappedOutputBuffers[bufferIndex] = mapInputResourceBitstreamBuffer.mappedResource;
}

void EncodeDecode::cpuWaitForFencePoint(ID3D12Fence* pFence, uint64_t nFenceValue)
{
    if (pFence->GetCompletedValue() < nFenceValue)
    {
        if (pFence->SetEventOnCompletion(nFenceValue, mEvent) != S_OK)
        {
            // Error
            // NVENC_THROW_ERROR("SetEventOnCompletion failed", NV_ENC_ERR_INVALID_PARAM);
            std::cout << "Error complete the event.\n";
        }

        WaitForSingleObject(mEvent, INFINITE);
    }
}

void EncodeDecode::waitForCompletionEvent(int eventIndex)
{
    // wait for 20s which is infinite on terms of gpu time
    if (WaitForSingleObject(mVPCompletionEvent[eventIndex], 20000) == WAIT_FAILED)
    {
        // NVENC_THROW_ERROR("Failed to encode frame", NV_ENC_ERR_GENERIC);
    }
}

/*
map resources into memory
set picture params
*/
NVENCSTATUS EncodeDecode::encodeFrameBuffer()
{
    NV_ENC_PIC_PARAMS picParams = {};

    mapEncoderResource(0);

    InterlockedIncrement(&mNOutputFenceVal);

    NV_ENC_INPUT_PTR inputBuffer = mVMappedInputBuffers[0];
    NV_ENC_OUTPUT_PTR outputBuffer = mVMappedOutputBuffers[0];

    mVInputRsrc[0]->pInputBuffer = inputBuffer;
    mVInputRsrc[0]->inputFencePoint.waitValue = mNInputFenceVal;
    mVInputRsrc[0]->inputFencePoint.bWait = true;

    mVOutputRsrc[0]->pOutputBuffer = outputBuffer;
    mVOutputRsrc[0]->outputFencePoint.signalValue = mNOutputFenceVal;
    mVOutputRsrc[0]->outputFencePoint.bSignal = true;

    // picture params
    picParams.version = NV_ENC_PIC_PARAMS_VER;
    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picParams.inputBuffer = mVInputRsrc[0];
    picParams.bufferFmt = mEBufferFormat;
    picParams.inputWidth = mWidth;
    picParams.inputHeight = mHeight;
    picParams.outputBitstream = mVOutputRsrc[0];
    picParams.completionEvent = mVPCompletionEvent[0];
    NVENCSTATUS nvStatus = mNVEnc.nvEncEncodePicture(mHEncoder, &picParams);

    waitForCompletionEvent(0); // wait for nvEncEncodePicture to finish
    // std::vector<uint8_t> mVEncodeOutData, containing bitstream data,
    // clear the previous encoded frame
    mVEncodeOutData.clear();
    // std::cout << "encode successfully!\n";

    /*
    after encoding, copy thing to bitstream (gpu)
    output data is frame encoded in bitstream
    */

    // NV_ENC_LOCK_BITSTREAM initialized to specify details about lock operation
    NV_ENC_LOCK_BITSTREAM lockBitstreamData = {NV_ENC_LOCK_BITSTREAM_VER};
    lockBitstreamData.outputBitstream = mVOutputRsrc[0];
    lockBitstreamData.doNotWait = false; // function should wait until the bitstream data is available.
    // nvEncLockBitstream lock the bitstream data associated with the encoder (mHEncoder)
    NVENC_API_CALL(mNVEnc.nvEncLockBitstream(mHEncoder, &lockBitstreamData));

    uint8_t* pData = (uint8_t*)lockBitstreamData.bitstreamBufferPtr; // The pointer pData is set to the start of the locked bitstream data
    // The bitstream data is then copied into a container (mVEncodeOutData)
    mVEncodeOutData.insert(mVEncodeOutData.end(), &pData[0], &pData[lockBitstreamData.bitstreamSizeInBytes]);
    // unlock a bitstream buffer that was previously locked for writing
    NVENC_API_CALL(mNVEnc.nvEncUnlockBitstream(mHEncoder, lockBitstreamData.outputBitstream));

    // write encoded frames to out_.h264
    if (outputEncodedFrames)
    {
        fpEncOut->write(reinterpret_cast<char*>(mVEncodeOutData.data()), mVEncodeOutData.size());
    }

    // The input resources (mVMappedInputBuffers[0]) used in the encoding process are unmapped
    NVENC_API_CALL(mNVEnc.nvEncUnmapInputResource(mHEncoder, mVMappedInputBuffers[0]));
    mVMappedInputBuffers[0] = nullptr;

    NVENC_API_CALL(mNVEnc.nvEncUnmapInputResource(mHEncoder, mVMappedOutputBuffers[0]));
    mVMappedOutputBuffers[0] = nullptr;

    // std::cout << "write bitstream successfully!\n";

    //printf("1");

    //decodeMutex = 0;

    return nvStatus;
}

void EncodeDecode::makeDefaultDecodingParams(CUVIDDECODECREATEINFO* pInitializeParams)
{
    pInitializeParams->CodecType = cudaVideoCodec_H264; // cudaVideoCodec_HEVC= h265, cudaVideoCodec_H264 = h264
    pInitializeParams->ChromaFormat = cudaVideoChromaFormat_420;
    pInitializeParams->OutputFormat = cudaVideoSurfaceFormat_NV12;
    pInitializeParams->bitDepthMinus8 = 0;
    pInitializeParams->DeinterlaceMode = cudaVideoDeinterlaceMode_Bob;
    pInitializeParams->ulNumOutputSurfaces = 1;
    pInitializeParams->ulCreationFlags = cudaVideoCreate_PreferCUVID;
    pInitializeParams->ulNumDecodeSurfaces = 4; // match ulMaxNumDecodeSurfaces
    pInitializeParams->vidLock = mCtxLock;
    pInitializeParams->ulWidth = mWidth;
    pInitializeParams->ulHeight = mHeight;
    pInitializeParams->ulMaxWidth = mWidth;
    pInitializeParams->ulMaxHeight = mHeight;
    pInitializeParams->ulTargetWidth = mWidth;
    pInitializeParams->ulTargetHeight = mHeight;
    pInitializeParams->enableHistogram = 0;
}

void EncodeDecode::initDecoder()
{
    // Initialise CUDA first
    // check how many gpus we have, use the first one
    cuInit(0);
    // Get the first GPU listed (change this for multi-GPU setups)
    int iGpu = 0;
    int nGpu = 0;
    cuDeviceGetCount(&nGpu);
    if (iGpu < 0 || iGpu >= nGpu)
    {
        std::ostringstream err;
        err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        throw std::invalid_argument(err.str());
    }
    CUdevice cuDevice = 0;
    CUresult deviceError = cuDeviceGet(&cuDevice, iGpu);
    char szDeviceName[80];
    cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
    std::cout << "GPU in use: " << szDeviceName << std::endl;

    cuDeviceGet(&mCudaDevice, 0);
    // Make the Cuda context
    CUresult contextError = cuCtxCreate(&mCudaContext, CU_CTX_SCHED_BLOCKING_SYNC, mCudaDevice);

    NVDEC_API_CALL(cuvidCtxLockCreate(&mCtxLock, mCudaContext));

    cuStreamCreate(&mCuvidStream, CU_STREAM_DEFAULT);

    // Make decoding parser
    /*
    a parser is a component responsible for analyzing the
    bitstream or input data and extracting relevant information or structures

    parse the bitstream and perform decoding
    */
    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = cudaVideoCodec_H264; // cudaVideoCodec_HEVC, cudaVideoCodec_H264
    videoParserParameters.ulMaxNumDecodeSurfaces = 4; // >3 works, number of surfaces (decoded frames) in parser’s DPB (decode picture
                                                      // buffer)
    videoParserParameters.ulClockRate = 0;
    videoParserParameters.ulMaxDisplayDelay = 0; // 0 = no delay
    videoParserParameters.pUserData = this;
    videoParserParameters.pfnSequenceCallback = nullptr;
    videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc; // called once per frame
    videoParserParameters.pfnDisplayPicture = nullptr;
    videoParserParameters.pfnGetOperatingPoint = nullptr;
    videoParserParameters.pfnGetSEIMsg = nullptr;
    NVDEC_API_CALL(cuvidCreateVideoParser(&mHParser, &videoParserParameters));

    // Check the decoding capabilities
    CUVIDDECODECAPS decodecaps;
    memset(&decodecaps, 0, sizeof(decodecaps));

    decodecaps.eCodecType = cudaVideoCodec_H264; // cudaVideoCodec_HEVC, cudaVideoCodec_H264
    decodecaps.eChromaFormat = cudaVideoChromaFormat_420;
    decodecaps.nBitDepthMinus8 = 0;

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext));
    NVDEC_API_CALL(cuvidGetDecoderCaps(&decodecaps));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

    if (!decodecaps.bIsSupported)
    {
        printf("Codec not supported on this GPU\n");
    }

    if ((mWidth > decodecaps.nMaxWidth) || (mHeight > decodecaps.nMaxHeight))
    {
        std::ostringstream errorString;
        errorString << std::endl
                    << "Resolution          : " << mWidth << "x" << mHeight << std::endl
                    << "Max Supported (wxh) : " << decodecaps.nMaxWidth << "x" << decodecaps.nMaxHeight << std::endl
                    << "Resolution not supported on this GPU";

        const std::string cErr = errorString.str();
        std::cout << cErr;
    }

    // Make the decoder
    CUVIDDECODECREATEINFO videoDecodeCreateInfo = {0};

    // makeDefaultDecodingParams(&videoDecodeCreateInfo);

    /*   m_displayRect.b = reconfigParams.display_area.bottom;
       m_displayRect.t = reconfigParams.display_area.top;
       m_displayRect.l = reconfigParams.display_area.left;
       m_displayRect.r = reconfigParams.display_area.right;*/

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext));
    NVDEC_API_CALL(cuvidCreateDecoder(&mHDecoder, &videoDecodeCreateInfo)); // create decoder
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(nullptr));
    presenterPtr = new FramePresenterD3D11(mCudaContext, mWidth, mHeight);
    makeDecoderOutputBuffers(); // allocate cuda memory
}

void EncodeDecode::makeDecoderOutputBuffers()
{
    CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr*)&mPDecoderFrame, getDecoderFrameSize()));
    CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr*)&mPDecoderRGBAFrame, mWidth * mHeight * 4));

    mPHostRGBAFrame = new uint8_t[mWidth * mHeight * 4];
}

/*
called by us once per frame buffer
perform parsing
trigger handlePicutreDecode to perform decoding
*/
void EncodeDecode::decodeFrameBuffer()
{
    static int bitstream_size = 0;
    CUVIDSOURCEDATAPACKET packet = {0};
    packet.payload = mVEncodeOutData.data();
    packet.payload_size = mVEncodeOutData.size();
    // bitstream_size += mVEncodeOutData.size();
    // std::cout << "bitstream_size " << bitstream_size << "\n";
    NVDEC_API_CALL(cuvidParseVideoData(mHParser, &packet));
}

void InitializeImageData(uint8_t* imageData, int width, int height)
{
    for (int i = 0; i < width * height * 4; i += 4)
    {
        // Set alpha channel to fully opaque
        imageData[i + 3] = 255; // Opaque (0xFF)
    }
}

int EncodeDecode::handlePictureDecode(CUVIDPICPARAMS* pPicParams)
{
    static int count = 0;
    // We have parsed an entire frame! Now let's decode it
    // std::cout << "Frame found: " << count++ << "\n\n";

    //  A context represents the environment in which CUDA operations and computations take place
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext)); // CUDA contexts are used to manage the state of the CUDA runtime
    NVDEC_API_CALL(cuvidDecodePicture(mHDecoder, pPicParams));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

    // provide information about a decoded picture
    CUVIDPARSERDISPINFO dispInfo;
    memset(&dispInfo, 0, sizeof(dispInfo));
    dispInfo.picture_index = pPicParams->CurrPicIdx;
    dispInfo.progressive_frame = !pPicParams->field_pic_flag;
    dispInfo.top_field_first = pPicParams->bottom_field_flag ^ 1;
    CUVIDGETDECODESTATUS DecodeStatus;              // CUVIDGETDECODESTATUS query the decode status of the CUDA Video Decoder
    memset(&DecodeStatus, 0, sizeof(DecodeStatus)); // initialize CUVIDGETDECODESTATUS with 0

    CUVIDPROCPARAMS videoProcessingParameters = {};
    videoProcessingParameters.progressive_frame = dispInfo.progressive_frame;
    videoProcessingParameters.second_field = dispInfo.repeat_first_field + 1;
    videoProcessingParameters.top_field_first = dispInfo.top_field_first;
    videoProcessingParameters.unpaired_field = dispInfo.repeat_first_field < 0;
    videoProcessingParameters.output_stream = mCuvidStream;

    CUdeviceptr dpSrcFrame = 0;
    unsigned int nSrcPitch = 0;
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext));
    NVDEC_API_CALL(cuvidMapVideoFrame(mHDecoder, dispInfo.picture_index, &dpSrcFrame, &nSrcPitch, &videoProcessingParameters));

    // Check we have decoded the frame correctly
    CUresult result = cuvidGetDecodeStatus(mHDecoder, dispInfo.picture_index, &DecodeStatus);

    if (result == CUDA_SUCCESS &&
        (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error || DecodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed))
    {
        std::cout << "Decode Error occurred for picture\n";
    }

    // CUDA_MEMCPY2D a 2D memory copy operation in CUDA
    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;                  // source memory is located on the GPU
    m.srcDevice = dpSrcFrame;                                // assign a pointer to the source GPU memory
    m.srcPitch = nSrcPitch;                                  // pitch (or width) of the source memory in bytes
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;                  // destination memory is located on the GPU
    m.dstDevice = (CUdeviceptr)(m.dstHost = mPDecoderFrame); // assign the destination device pointer
    m.dstPitch = mWidth;
    m.WidthInBytes = mWidth;
    m.Height = mHeight;
    // cuMemcpy2DAsync(&m, mCuvidStream)
    CUDA_DRVAPI_CALL(cuMemcpy2D(&m)); // CUDA function for asynchronously copying memory between two 2D memory regions
    // cudaStreamSynchronize
    //  TODO: why update the memory copy operation?
    m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((mHeight + 1) & ~1)); // updated with a new source device pointer
    m.dstDevice = (CUdeviceptr)(m.dstHost = mPDecoderFrame + m.dstPitch * mHeight);
    m.Height = (int)(ceil(mHeight * 0.5));

    CUDA_DRVAPI_CALL(cuMemcpy2D(&m));

    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    NVDEC_API_CALL(cuvidUnmapVideoFrame(mHDecoder, dpSrcFrame));

    // perform color format conversion
    // convert video frame data from the NV12 format to the BGRA32 format
    // NV12 is a commonly used video format where the Y (luminance) and UV (chrominance) components are stored in separate planes.
    // mPDecoderFrame: pointer to input frame data
    // mPDecoderRGBAFrame: Pointer to the output buffer where the converted BGRA32 data will be stored,
    // data type is cast to uint8_t*, indicating that the output data is treated as an array of bytes
    // mPDecoderRGBAFrame would represent the frame in BGRA32 format
    // mWidth * 4: The pitch (stride) of the output data
    Nv12ToColor32<BGRA32>(mPDecoderFrame, mWidth, (uint8_t*)mPDecoderRGBAFrame, mWidth * 4, mWidth, mHeight);

    // code copy from cuda (gpu) to cpu
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext)); // push the CUDA context (mCudaContext) onto the current thread's CUDA context stack.
    //  copying data from mPDecoderRGBAFrame (GPU device memory) to mPHostRGBAFrame (CPU host memory)
    CUDA_DRVAPI_CALL(cuMemcpyDtoH(mPHostRGBAFrame, mPDecoderRGBAFrame, mWidth * mHeight * 4));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL)); // o pop the current CUDA context off the CUDA context stack to release resources

    // show mPDecoderRGBAFrame , access cuda memory from directx12 is difficult
    // create directx12 texture, register with cuda xxx, write to this texture
    // don't need to transfer from cuda to cpu and to directx12
    // do: make the memory using directx12, write to the resource


    // FramePresenterD3D11 presenter(mCudaContext, mWidth, /*mHeight);
    presenterPtr->PresentDeviceFrame((uint8_t*)mPDecoderRGBAFrame, mWidth * 4, 0);

    //printf("3");

    return 0;
}

int EncodeDecode::getDecoderFrameSize()
{
    float chromaHeightFactor = 0.5;
    int lumaHeight = mHeight;
    int chromaHeight = (int)(ceil(lumaHeight * chromaHeightFactor));
    int nChromaPlanes = 1;
    int nBPP = 1;
    int result = mWidth * (lumaHeight + (chromaHeight * nChromaPlanes)) * nBPP;

    return result;
}

void EncodeDecode::loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo)
{
    mpScene = Scene::create(getDevice(), path);
    mpCamera = mpScene->getCamera();

    // mpScene->setCameraController(Scene::CameraControllerType::Orbiter);
    // mpCamCtrl = std::make_unique<OrbiterCameraController>(mpCamera);

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    std::cout << '\n';
    std::cout << "pTargetFbo Width: " << (float)pTargetFbo->getWidth() << std::endl;
    std::cout << "pTargetFbo Height: " << (float)pTargetFbo->getHeight() << std::endl;
    std::cout << '\n';

    // Get shader modules and type conformances for types used by the scene.
    // These need to be set on the program in order to use Falcor's material system.
    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    // Get scene defines. These need to be set on any program using the scene.
    auto defines = mpScene->getSceneDefines();

    // Create raster pass.
    // This utility wraps the creation of the program and vars, and sets the necessary scene defines.
    ProgramDesc rasterProgDesc;
    rasterProgDesc.addShaderModules(shaderModules);
    rasterProgDesc.addShaderLibrary("Samples/EncodeDecode/EncodeDecode.3d.slang").vsEntry("vsMain").psEntry("psMain");
    rasterProgDesc.addTypeConformances(typeConformances);

    // rasterpass
    mpRasterPass = RasterPass::create(getDevice(), rasterProgDesc, defines);

    // We'll now create a raytracing program. To do that we need to setup two things:
    // - A program description (ProgramDesc). This holds all shader entry points, compiler flags, macro defintions,
    // etc.
    // - A binding table (RtBindingTable). This maps shaders to geometries in the scene, and sets the ray generation and
    // miss shaders.
    //
    // After setting up these, we can create the Program and associated RtProgramVars that holds the variable/resource
    // bindings. The Program can be reused for different scenes, but RtProgramVars needs to binding table which is
    // Scene-specific and needs to be re-created when switching scene. In this example, we re-create both the program
    // and vars when a scene is loaded.

    ProgramDesc rtProgDesc;
    rtProgDesc.addShaderModules(shaderModules);
    rtProgDesc.addShaderLibrary("Samples/EncodeDecode/EncodeDecode.rt.slang");
    rtProgDesc.addTypeConformances(typeConformances);
    rtProgDesc.setMaxTraceRecursionDepth(3); // 1 for calling TraceRay from RayGen, 1 for calling it from the
                                             // primary-ray ClosestHit shader for reflections, 1 for reflection ray
                                             // tracing a shadow ray
    rtProgDesc.setMaxPayloadSize(24);        // The largest ray payload struct (PrimaryRayData) is 24 bytes. The payload size
                                             // should be set as small as possible for maximum performance.

    ref<RtBindingTable> sbt = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
    sbt->setRayGen(rtProgDesc.addRayGen("rayGen"));
    sbt->setMiss(0, rtProgDesc.addMiss("primaryMiss"));
    sbt->setMiss(1, rtProgDesc.addMiss("shadowMiss"));
    auto primary = rtProgDesc.addHitGroup("primaryClosestHit", "primaryAnyHit");
    auto shadow = rtProgDesc.addHitGroup("", "shadowAnyHit");
    sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), primary);
    sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), shadow);

    mpRaytraceProgram = Program::create(getDevice(), rtProgDesc, defines);
    mpRtVars = RtProgramVars::create(getDevice(), mpRaytraceProgram, sbt);
}

void EncodeDecode::setPerFrameVars(const Fbo* pTargetFbo)
{
    auto var = mpRtVars->getRootVar();
    var["PerFrameCB"]["invView"] = inverse(mpCamera->getViewMatrix());
    var["PerFrameCB"]["viewportDims"] = Falcor::float2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
    float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
    var["PerFrameCB"]["tanHalfFovY"] = std::tan(fovY * 0.5f);
    var["PerFrameCB"]["sampleIndex"] = mSampleIndex++;
    var["PerFrameCB"]["useDOF"] = mUseDOF;
    var["gOutput"] = mpRtOut;
}


void EncodeDecode::setBitRate(unsigned int br)
{
    bitRate = br; // Assign the private member
    std::cout << "setBitRate  " << bitRate << "\n";
}

// void EncodeDecode::setFrameRate(unsigned int fps)
// {
//     frameRate = fps;
//     // fast 34, normal 68, slow 68
//     frameLimit = 30 * frameRate / 30.0; // 34, 68, 30
//     // frameLimit = 135; // TODO: change to above line
//     std::cout << "setFrameRate  " << frameRate << "/n";
// }

void EncodeDecode::setFrameRate(unsigned int fps)
{
    frameRate = fps; // Assign the private member
    frameLimit = frameRate + numOfFrames * frameRate / 30.0; // 68, 34, 45, 30
    std::cout << "setFrameRate  " << frameRate << "\n";
    std::cout << "frameLimit  " << frameLimit << "\n";
}


void EncodeDecode::setDefaultScene(std::string scenePath)
{
    strcpy(kDefaultScene, scenePath.c_str());
    std::cout << "kDefaultScene  " << kDefaultScene << "\n";
}



void EncodeDecode::setSpeed(unsigned int input)
{
    speed = input;
    std::cout << "setSpeed  " << speed << "\n";
}

void EncodeDecode::setSceneName(std::string scene)
{
    sceneName = scene;
    std::cout << "sceneName  " << sceneName << "\n";
}


void EncodeDecode::setRefPrefix(std::string scene, unsigned int speedInput)
{
    std::string fullPath = std::string(szRefPrefixFilePath) + scene + std::to_string(speedInput) + "/";

    // Ensure the result fits into the original char array
    if (fullPath.length() < sizeof(szRefPrefixFilePath)) {
        strcpy(szRefPrefixFilePath, fullPath.c_str());
    } else {
        std::cerr << "Error: Resulting string is too long for the buffer." << std::endl;
    }
    std::cout << "setRefPrefix  " << szRefPrefixFilePath << "\n";

    std::filesystem::path dirPath(fullPath);

    if (!std::filesystem::exists(dirPath)) {
        // Create the directory if it does not exist
        try {
            if (std::filesystem::create_directories(dirPath)) {
                std::cout << "Directory created: " << dirPath << std::endl;
            } else {
                std::cout << "Directory already exists or cannot be created: " << dirPath << std::endl;
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else {
        std::cout << "Directory already exists: " << dirPath << std::endl;
    }
}


void EncodeDecode::setMotionPrefix(std::string scene, unsigned int speedInput, unsigned int framerate, unsigned int bitrate, unsigned int height)
{

    fs::path dirPath = fs::path(szMotionPrefixFilePath) /
                        (scene + "_" + std::to_string(speedInput)) /
                       (std::to_string(bitrate) + "kbps") /
                       ("fps" + std::to_string(framerate)) /
                       (std::to_string(framerate) + "_" + std::to_string(height) + "_" + std::to_string(bitrate));

    try {
        // Create directories
        if (fs::create_directories(dirPath)) {
            std::cout << "Successfully created directory: " << dirPath << std::endl;
        } else {
            std::cout << "Directory already exists or failed to create: " << dirPath << std::endl;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    std::string dirPathStr = dirPath.string();
    // if (dirPathStr.size() >= bufferSize) {
    //     std::cerr << "Error: Directory path is too long to fit in the provided buffer" << std::endl;
    //     return;
    // }

    // Copy the path to the char array
    std::strncpy(szMotionPrefixFilePath, dirPathStr.c_str(), sizeof(szMotionPrefixFilePath));
    szMotionPrefixFilePath[sizeof(szMotionPrefixFilePath) - 1] = '\0'; // Ensure null-termination


    // std::string fullPath = std::string(szMotionPrefixFilePath) + scene + std::to_string(speedInput) + "/";

    // // Ensure the result fits into the original char array
    // if (fullPath.length() < sizeof(szMotionPrefixFilePath)) {
    //     strcpy(szMotionPrefixFilePath, fullPath.c_str());
    // } else {
    //     std::cerr << "Error: Resulting string is too long for the buffer." << std::endl;
    // }
    // std::cout << "setMotionPrefix  " << szMotionPrefixFilePath << "\n";
    // std::cout << "frameRate  " << frameRate << "\n";
    // std::cout << "bitRate  " << bitRate << "\n";

    // std::filesystem::path dirPath(fullPath);

    // if (!std::filesystem::exists(dirPath)) {
    //     // Create the directory if it does not exist
    //     try {
    //         if (std::filesystem::create_directories(dirPath)) {
    //             std::cout << "Directory created: " << dirPath << std::endl;
    //         } else {
    //             std::cout << "Directory already exists or cannot be created: " << dirPath << std::endl;
    //         }
    //     } catch (const std::filesystem::filesystem_error& e) {
    //         std::cerr << "Error: " << e.what() << std::endl;
    //     }
    // } else {
    //     std::cout << "Directory already exists: " << dirPath << std::endl;
    // }
}





void EncodeDecode::renderRaster(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "renderRaster");

    mpRasterPass->getState()->setFbo(pTargetFbo);
    mpScene->rasterize(pRenderContext, mpRasterPass->getState().get(), mpRasterPass->getVars().get());
}

void EncodeDecode::createMipMaps(RenderContext* pRenderContext)
{
    mpRtMip->generateMips(pRenderContext, false);
    uint32_t numMips = mpRtMip->getMipCount();
}


void EncodeDecode::renderRT(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo, int fCount)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "renderRT");

    setPerFrameVars(pTargetFbo.get());

    // comment below line to fix artifacts of decoded frames
    // pRenderContext->clearUAV(mpRtOut->getUAV().get(), kClearColor); // kClearColorb is green
    mpScene->raytrace(pRenderContext, mpRaytraceProgram.get(), mpRtVars, Falcor::uint3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1));
    /*
    *
    performing a blit operation from the Shader Resource View (mpRtOut->getSRV())
    to the render target view (pTargetFbo->getRenderTargetView(0)) using (pRenderContext).

    blit：block transfer, in graphics means copy pixel data from A to B
    render target: destination for the graphical information produced by the rendering pipeline,
    a buffer where the rendering output is directed, e.g. framebuffer, texture
    Shader Resource View (SRV): a way to expose a texture to shaders for reading
    A render target view is a way to expose a texture to shaders for writing
    */

    // mpRtOut and pTargetFbo have the same size, i.e. width height of reference frames
    //pRenderContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));

    // std::cout << "count fCount " << fCount << "\n";


    pRenderContext->signal(mpDecodeFence.get(), mNDecodeFenceVal);
    pRenderContext->blit(motionVectorTexture->getSRV(), mpRtMip->getRTV());
    createMipMaps(pRenderContext);
    // TODO: change mip level
    // This function is designed to read data from a specific subresource of a GPU texture
    // (e.g., a texture that has multiple layers, or mip levels) and return the data in a std::vector<uint8_t> for CPU access
    std::vector<uint8_t> val = pRenderContext->readTextureSubresource(mpRtMip.get(), mipLevels-1);
    // std::cout << "renderRT mipLevels: " << mipLevels << std::endl;
    // std::cout << "renderRT frameRate: " << frameRate << std::endl;
    // std::cout << "renderRT bitRate: " << bitRate << std::endl;
  /*  std::cout << "Number of elements in level 10: " << val.size() << std::endl;
    std::cout << "Number of elements in level 0: " << pRenderContext->readTextureSubresource(mpRtMip.get(), 0).size() << std::endl;*/
    float* t = (float*)val.data();
    // The length will be the number of float elements in the data.
    std::size_t length = val.size() / sizeof(float); // val.size() is total bytes, sizeof(float) is 4 bytes.
    // float t1 = t[0] * mWidth;
    // float t2 = t[1] * mHeight;
    float t1 = t[0];
    float t2 = t[1];
    // std::cout << "t1 " << t1 << "\n"; // Print as integer
    // std::cout << "length " << length << "\n"; // Print as integer
    // std::cout << "t2 " << t2 << "\n"; // Print as integer

    // double hypotenuse = sqrt(t1 * t1 + t2 * t2);
    // double velocity = frameRate * hypotenuse ;
    //std::cout << "The hypotenuse of the right triangle is: " << hypotenuse << "\n";
    // std::cout << "v: " << velocity << "\n";
    // std::cout << "val.size(): " << val.size() << "\n";
    // std::cout << "================== oss\n";


    if (fCount >= frameRate)
    {
        // comment out to avoid saving motion vector
        std::ostringstream oss;  // Create a string stream to construct the file name
        // oss << "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/motion/"
        oss << szMotionPrefixFilePath << "/"
            << fCount << "_" << frameRate << "_" << mHeight << "_"
            // << bitRate/1000 << "_" << "_" << sceneName << "_" << speed << ".dat";  // Append details
            << bitRate/1000 << "_" << speed << ".dat";  // Append details
        std::string fileName = oss.str();

        // save to DAT file
        try {
            saveVectorToFile(val, fileName);
            // std::cout << "File saved successfully." << std::endl;
        } catch (const std::ios_base::failure& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }


    // std::ofstream outFile(fileName, std::ios::out | std::ios::app);
    // // std::ofstream outFile(fileName, std::ios::app); // Open the file in append mode

    // // if (!outFile.is_open())
    // // {
    // //     std::cerr << "Failed to open file: " << fileName << std::endl;
    // // }
    // // outFile << "t0, t1, v: " << t1 << " " << t2 << " " << velocity << " frame " << fCount << std::endl;
    // outFile << "t0, t1, v: " << t1 << " " << t2 << " frame " << fCount << std::endl;

    // // Logging the elements
    // if (!val.empty()) {
    //     // outFile.write(reinterpret_cast<const char*>(val.data()), val.size());
    //     outFile.write(reinterpret_cast<const char*>(val.data()), val.size() * sizeof(float));

    // }

    // // binary file
    // if (!val.empty()) {
    //     outFile.write(reinterpret_cast<const char*>(val.data()), val.size());
    // }

    // outFile << fCount << " " << velocity << std::endl;
    // outFile.close();


    //// write to bmp file
    // if (outputReferenceFrames)
    //{
    //     static int fCount = 0;
    //     snprintf(szRefOutFilePath, sizeof(szRefOutFilePath), "%s%d.bmp", refBaseFilePath, fCount++);
    //     // auto fileformat = Bitmap::getFormatFromFileExtension(szRefOutFilePath);
    //     mpRtOut->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);

    //}
}


// log motion vector into dat file
int runMain(int argc, char** argv)
{
    unsigned int bitrate = std::stoi(argv[1]);
    unsigned int framerate = std::stoi(argv[2]);
    unsigned int width = std::stoi(argv[3]);
    unsigned int height = std::stoi(argv[4]);
    std::string scene = argv[5];
    unsigned int speedInput = std::stoi(argv[6]);
    std::string scenePath = argv[7];


    // unsigned int width = 1920;
    // unsigned int height = 1080;
    // unsigned int bitrate = 8000;
    // unsigned int framerate = 30;

    // std::string scene = "suntemple_statue";
    // unsigned int speedInput = 1;
    // std::string scenePath = "suntemple_statue/path1_seg1.fbx";

    std::cout << "\n\nframerate runmain  " << framerate << "\n";
    std::cout << "bitrate runmain  " << bitrate << "\n";
    std::cout << "width runmain  " << width << "\n";
    std::cout << "height runmain  " << height << "\n";
    std::cout << "scene " << scene << std::endl;
    std::cout << "speed " << speedInput << std::endl;
    std::cout << "scenePath " << scenePath << std::endl;


    SampleAppConfig config;
    config.windowDesc.title = "EncodeDecode";
    config.windowDesc.resizableWindow = true;
    config.colorFormat = ResourceFormat::BGRA8Unorm;
    config.windowDesc.width = width;
    config.windowDesc.height = height;

    EncodeDecode encodeDecode(config);
    encodeDecode.setBitRate(bitrate * 1000);
    encodeDecode.setFrameRate(framerate);
    encodeDecode.setDefaultScene(scenePath);
    encodeDecode.setMotionPrefix(scene, speedInput, framerate, bitrate, height);
    // encodeDecode.setRefPrefix(scene, speedInput);
    encodeDecode.setSpeed(speedInput);
    encodeDecode.setSceneName(scene);

    return encodeDecode.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
