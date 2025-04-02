// EncodeDecode.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once

#include <filesystem>
#include <iostream>
// #include <curl/curl.h>
#include <d3d12.h>
#include <d3d11.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <iomanip>
#include <random>
#include <cppcodec/base64_rfc4648.hpp>

#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/TextRenderer.h"
#include "Core/API/NativeHandleTraits.h"
#include "RenderGraph/RenderGraph.h"

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
std::chrono::steady_clock::time_point last_send_time = std::chrono::steady_clock::now();


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

#define NVENC_THROW_ERROR( errorStr, errorCode )                                                         \
    do                                                                                                   \
    {                                                                                                    \
        throw NVENCException::makeNVENCException(errorStr, errorCode, __FUNCTION__, __FILE__, __LINE__); \
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

std::pair<uint32_t, uint32_t> getRandomStartCoordinates(int frameWidth, int frameHeight, int patchWidth, int patchHeight) {
    // Random number generation setup
    std::random_device rd;  // Seed
    std::mt19937 gen(rd());  // Mersenne Twister generator
    std::uniform_int_distribution<> distribX(0, frameWidth - patchWidth);
    std::uniform_int_distribution<> distribY(0, frameHeight - patchHeight);

    // Generate random startX and startY within valid bounds
    uint32_t startX = distribX(gen);
    uint32_t startY = distribY(gen);

    return {startX, startY};
}


// Encode the patch data as base64 string
std::string base64_encode_patch(const std::vector<uint8_t>& patchData) {
    return cppcodec::base64_rfc4648::encode(patchData);
}

// // Function to asynchronously send patch and velocity to the HTTP server
// // const std::vector<std::vector<float>>& patch
// void send_http_request_async(const std::vector<uint8_t>& patchData, float velocity) {
//     std::thread([patchData, velocity]() {
//         auto start = std::chrono::high_resolution_clock::now();

//         CURLM* multi_handle;
//         CURL* easy_handle;
//         // When still_running becomes 0, it means all HTTP requests have completed, and you can proceed with any post-processing steps or cleanup.
//         int still_running = 0; // Number of active transfers

//         // Create a JSON object with patch and velocity
//         nlohmann::json j;
//         // Base64 encode the patch data and add to the JSON
//         std::string encodedPatch = base64_encode_patch(patchData);
//         j["patch"] = encodedPatch;
//         j["velocity"] = velocity;
//         // std::cout << "patch \n " << encodedPatch << "\n";
//         std::cout << "JSON payload:\n" << j.dump(4) << "\n";  // Pretty-print JSON with 4-space indentation


//         // Convert JSON to string
//         std::string json_str = j.dump();
//         // Initialize curl easy and multi handles
//         easy_handle = curl_easy_init();
//         multi_handle = curl_multi_init();

//         if (easy_handle && multi_handle) {
//             // Set URL and POST data
//             curl_easy_setopt(easy_handle, CURLOPT_URL, "http://localhost:8000/predict");
//             curl_easy_setopt(easy_handle, CURLOPT_POSTFIELDS, json_str.c_str());

//             // Set HTTP headers
//             struct curl_slist* headers = NULL;
//             headers = curl_slist_append(headers, "Content-Type: application/json");
//             curl_easy_setopt(easy_handle, CURLOPT_HTTPHEADER, headers);
//             // Add easy handle to multi handle
//             curl_multi_add_handle(multi_handle, easy_handle);

//             auto now = std::chrono::system_clock::now();
//             std::time_t currentTime = std::chrono::system_clock::to_time_t(now); // getting calendar time
//             std::tm* localTime = std::localtime(&currentTime);
//             std::cout << "Current local time: " << std::put_time(localTime, "%Y-%m-%d %H:%M:%S") << std::endl;
//             auto stop1 = std::chrono::high_resolution_clock::now();
//             std::chrono::duration<double> elapsed_seconds_stop0 = stop1 - start;
//             std::cout << "Before send patch takes: " << elapsed_seconds_stop0.count() << " seconds\n";

//             // Perform the request asynchronously
//             curl_multi_perform(multi_handle, &still_running);

//             // Polling loop to check the status of the request
//             while (still_running) {
//                 int numfds;
//                 curl_multi_wait(multi_handle, NULL, 0, 1000, &numfds);  // Wait for data/events
//                 curl_multi_perform(multi_handle, &still_running);       // Perform any outstanding transfers
//             }

//             auto end = std::chrono::high_resolution_clock::now();
//             std::chrono::duration<double> elapsed_seconds_stop1 = end - stop1;
//             std::chrono::duration<double> elapsed_seconds = end - start;
//             std::cout << "Request round trip: " << elapsed_seconds_stop1.count() << " seconds\n";
//             std::cout << "Request completed in: " << elapsed_seconds.count() << " seconds\n";


//             curl_multi_remove_handle(multi_handle, easy_handle);
//             curl_easy_cleanup(easy_handle);
//             curl_multi_cleanup(multi_handle);
//         }
//     }).detach(); // Detach the thread to run independently
// }


// void send_patch_if_needed() {
//     std::cout << "enter: send_patch_if_needed  " << "\n";
//     auto current_time = std::chrono::steady_clock::now();
//     std::chrono::duration<double> time_elapsed = current_time - last_send_time;
//     std::cout << "time_elapsed  " << time_elapsed.count() << "\n";

//     if (time_elapsed.count() >= 5.0) {
//         // Capture patch and velocity and send to the server
//         // std::vector<std::vector<float>> patch = capture_patch();  // Example capture function
//         float velocity = 0.75;  // Example velocity calculation
//         send_http_request_async(velocity);

//         // Update the last_send_time
//         last_send_time = current_time;
//     }
// }


// static const Falcor::float4 kClearColor(0.38f, 0.52f, 0.10f, 1);
// static const Falcor::float4 kClearColor(0.5f, 0.16f, 0.098f, 1);
static const Falcor::float4 kClearColor(1.f, 0.f, 0.0f, 1);


// constructor
EncodeDecode::EncodeDecode(const SampleAppConfig& config) : SampleApp(config)
{
    /*
    1422x800 dec not working
    new pairs: 1536, 1200; 864, 676
    */
    /* std::vector<unsigned int> bitrates = {3000, 4000, 5000};
    for (unsigned int bitrate : bitrates)
    {*/
        mWidth = config.windowDesc.width;   // 1920, 4096, 1280, 854, 640, 960, 1024, 1280, 1440, 1423
        mHeight = config.windowDesc.height; // 1080, 2160, 720, 480, 360, 540, 600, 800, 900, 800

        std::cout << '\n';
        std::cout << "mWidth: " << mWidth << std::endl;
        std::cout << "mHeight: " << mHeight << std::endl;
        std::cout << '\n';

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
            // mWidth, mHeight, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess |
            // ResourceBindFlags::ShaderResource
            //  TODO: change the width and height of the reference frame size // 1920, 1080, 854, 480
            mWidth, mHeight, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );

        std::cout << "bitrate: " << bitRate << std::endl;


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

    std::string log;

    if (getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false)
    {
        FALCOR_THROW("Device does not support raytracing!");
    }

    initEncoder();
    // initDecoder();
    std::cout << "load scene: " << std::endl;

    loadScene(kDefaultScene, getTargetFbo().get());

    Properties gBufferProps = {};
    Properties fXAAProps = {};

    // RenderGraph有DirectedGraph, DirectedGraph 存储了 PassId 和 EdgeId 的关系，
    // 而 RenderGraph 则存储了两个 Id 所指向的资源。而这些关系的产生都在 addPass 和 addEdge 中完成
    mpRenderGraph = RenderGraph::create(mpDevice, "EncodeDecode");
    mpRenderGraph->createPass("GBuffer", "GBufferRaster", gBufferProps);
    // mpRenderGraph->createPass("FXAA", "FXAA", fXAAProps);
    mpRenderGraph->createPass("TAA", "TAA", fXAAProps);

    mpRenderGraph->onResize(getTargetFbo().get());
    mpRenderGraph->setScene(mpScene);

    mpRenderGraph->addEdge("GBuffer.mvec", "TAA.motionVecs"); // source texture, output texture
    ////mpRenderGraph->markOutput("GBuffer.mvec"); // Mark a render pass output as the graph's output.

    mpRenderGraph->markOutput("TAA.colorOut");
    mpRenderGraph->setInput("TAA.colorIn", mpRtOut);

    mpRenderGraph->compile(pRenderContext, log);


    // allocate memory so encoder can work with what we need
    // one buffer each time
    makeEncoderInputBuffers(1);
    makeEncoderOutputBuffers(1);
}

/*
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
        // mpCamera->setPosition(Falcor::float3(0.107229, 1.44531, -1.40544));
        mpCamera->setFocalLength(18);
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }
}

// Function to extract a 128x128 patch from the rendered frame
// mpRenderGraph->getOutput("TAA.colorOut")->asTexture()
// frame is likely on gpu. pRenderContext: The rendering context. Use it to bind state and dispatch calls to the GPU
void EncodeDecode::extract_patch_from_frame(std::vector<uint8_t>& renderedFrameVal, uint32_t frameWidth, uint32_t frameHeight, uint32_t patchWidth, uint32_t patchHeight, std::vector<uint8_t>& patchData)
{
    // std::cerr << "frameWidth " << frameWidth << std::endl;
    // std::cerr << "frameHeight " << frameHeight << std::endl;
    // std::cerr << "patchWidth " << patchWidth << std::endl;
    // std::cerr << "patchHeight " << patchHeight << std::endl;
    uint32_t numChannels = 4;  // For BGRA8 format (8 bits per channel, 4 channels)
    auto [startX, startY] = getRandomStartCoordinates(frameWidth, frameHeight, patchWidth, patchHeight);
    // uint32_t startX = 0;
    // uint32_t startY = 0;
    std::cout << "Random startX: " << startX << ", startY: " << startY << std::endl;

    // Make sure the patch doesn't exceed the texture bounds
    if (startX + patchWidth > frameWidth || startY + patchHeight > frameHeight) {
        std::cerr << "Patch exceeds texture bounds!" << std::endl;
        return;
    }

    // Extract the patch
    for (uint32_t y = 0; y < patchHeight; ++y) {
        for (uint32_t x = 0; x < patchWidth; ++x) {
            uint32_t fullIndex = ((startY + y) * frameWidth + (startX + x)) * numChannels;
            uint32_t patchIndex = (y * patchWidth + x) * 3;
            // Convert from BGRA to RGBA by swapping the B and R channels
            patchData[patchIndex] = renderedFrameVal[fullIndex + 2];       // Red
            patchData[patchIndex + 1] = renderedFrameVal[fullIndex + 1];   // Green
            patchData[patchIndex + 2] = renderedFrameVal[fullIndex];       // Blue
        }
    }
}

// renderframe from sampleapp calls onFrameRender
void EncodeDecode::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{

    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    // std::cout << "mpCamera: (" << mpCamera->getPosition().x << ", " << mpCamera->getPosition().y << ", " << mpCamera->getPosition().z << ")\n";

    static double timeSecs = 0; // timeSecs is the time through animation, i.e. camera path
    if (mpScene)
    {
        Scene::UpdateFlags updates = mpScene->update(pRenderContext, speed * timeSecs); // 2* timesec, 0.5
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

        mpRenderGraph->execute(pRenderContext);
        pRenderContext->signal(mpDecodeFence.get(), mNDecodeFenceVal);

        // blit from one frame buffer (or texture) to another
        // important for displaying the final rendered image to the screen
        // framebuffer object (FBO) that represents the final render target, usually the screen or a backbuffer
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), pTargetFbo->getRenderTargetView(0));

        // // send_patch_if_needed();
        // if (fcount == 100) {
        //     ref<Texture> frameTexture = mpRenderGraph->getOutput("TAA.colorOut")->asTexture();
        //     uint32_t frameWidth = frameTexture->getWidth();
        //     uint32_t frameHeight = frameTexture->getHeight();
        //     // auto format = frameTexture->getFormat(); // BGRA8Unorm (47)
        //     uint32_t patchWidth = 128;
        //     uint32_t patchHeight = 128;

        //     std::vector<uint8_t> patchData = pRenderContext->readTexturePatch(frameTexture.get(), 0, 0, 128, 128);
        //     //std::vector<uint8_t> patchData(patchWidth * patchHeight * 3);
        //     //extract_patch_from_frame(renderedFrameVal, frameWidth, frameHeight, patchWidth, patchHeight, patchData);
        //     stbi_write_png("patch_output_rgba.png", patchWidth, patchHeight, 3, patchData.data(), patchWidth * 3);

        //     send_http_request_async(patchData, 0.75);
        //     // send_http_request_async(0.75);
        // }

        // // allocate memory so encoder can work with what we need
        // // one buffer each time
        // makeEncoderInputBuffers(1);
        // makeEncoderOutputBuffers(1);

        cpuWaitForFencePoint(mpDecodeFence->getNativeHandle().as<ID3D12Fence*>(), mNDecodeFenceVal);

        // if (outputReferenceFrames && (fCount_rt >= frameRate))
        if (outputReferenceFrames && (fCount_rt >= 1))
        // if (outputReferenceFrames)
        {
            // std::cout<< "fCount_rt-frameRate " << fCount_rt-frameRate << "\n";
            // snprintf(szRefOutFilePath, sizeof(szRefOutFilePath), "%s%d.bmp", refBaseFilePath, fCount_rt-frameRate);
            snprintf(szRefOutFilePath, sizeof(szRefOutFilePath), "%s%d.bmp", refBaseFilePath, fCount_rt);
            // mpRtOut->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);
            mpRenderGraph->getOutput("TAA.colorOut")->asTexture()->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);
            // ref<Texture> mpFXAA = mpRenderGraph->getOutput("FXAA.dst")->asTexture();
            // mpFXAA->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);
        }

        // if (fCount_rt > 0)  // 2
        if (fCount_rt >= 1)  // 2
        {
            // std::cout << "fCount_rt: " << fCount_rt << "\n";

            encodeFrameBuffer(); // write encoded data into h264
            // decodeFrameBuffer();

            if (outputDecodedFrames)
            {
                snprintf(szDecOutFilePath, sizeof(szDecOutFilePath), "%s%d.bmp", decBaseFilePath, fcount);
                writeBMP(szDecOutFilePath, mPHostRGBAFrame, mWidth, mHeight);
            }

            if (frameLimit > 0 && fcount >= frameLimit)
            {
                std::exit(0);
            }
            timeSecs += 1.0 / frameRate; // disable line 520 about update timeSecs
        }
        fCount_rt += 1;
        ++fcount;
        // std::cout << "fcount " << fcount << "\n";
        // timeSecs += 1.0 / frameRate; // disable line 515 about update timeSecs
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
    // pIntializeParams->encodeConfig->rcParams.targetQuality = 50; // NV_ENC_PARAMS_RC_CONSTQP, NV_ENC_PARAMS_RC_CBR,
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
        // NV_ENC_CODEC_HEVC_GUID NV_ENC_CODEC_H264_GUID
        &mEncoderInitializeParams, NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY
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
    // mEncodeConfig.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH; // TODO: h264 only, add h265
    mEncodeConfig.encodeCodecConfig.hevcConfig.idrPeriod = NVENC_INFINITE_GOPLENGTH;


    // set bitrate 500000 (low quality) 1000000 1200000 (1200 - standard definition)
    // 3000000 4000000 (4000 - hd) 5000000, 8000000 (full hd) 10000000 15 Mbps - 30 Mbps 30000000 (4k)
    mEncodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR; // NV_ENC_PARAMS_RC_VBR NV_ENC_PARAMS_RC_CONSTQP, NV_ENC_PARAMS_RC_CBR,
    mEncodeConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION; // not valid for h264
    // disable below for CRF (target quality)
    mEncodeConfig.rcParams.averageBitRate = bitRate;
    // mEncodeConfig.rcParams.averageBitRate = 0;
    mEncodeConfig.rcParams.maxBitRate = bitRate;
    mEncodeConfig.rcParams.vbvBufferSize =
        // (mEncodeConfig.rcParams.averageBitRate * mEncoderInitializeParams.frameRateDen / mEncoderInitializeParams.frameRateNum) * 5;
        (mEncodeConfig.rcParams.maxBitRate * mEncoderInitializeParams.frameRateDen / mEncoderInitializeParams.frameRateNum) * 5;
    // mEncodeConfig.rcParams.maxBitRate = mEncodeConfig.rcParams.averageBitRate;
    mEncodeConfig.rcParams.vbvInitialDelay = mEncodeConfig.rcParams.vbvBufferSize;

    std::cout << "\naverageBitRate " << mEncodeConfig.rcParams.averageBitRate << "\n";
    std::cout << "\nmaxBitRate " << mEncodeConfig.rcParams.maxBitRate << "\n";

    NVENC_API_CALL(mNVEnc.nvEncInitializeEncoder(mHEncoder, &mEncoderInitializeParams));

    mNEncoderBuffer = mEncodeConfig.frameIntervalP + mEncodeConfig.rcParams.lookaheadDepth;

    mEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    mVPCompletionEvent.resize(1, nullptr);

    for (uint32_t i = 0; i < mVPCompletionEvent.size(); i++)
    {
        mVPCompletionEvent[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
    }


    if (outputEncodedFrames)
        {
            // generate outputfile name with timestamp
            std::ostringstream newFilePath;
            std::time_t t = std::time(nullptr);
            std::tm tm = *std::localtime(&t);
            char dateStr[11];
            std::strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", &tm);
            // Remove extension and replace slashes with underscores in scenePath, sceneNameOnly e.g. path1_seg1
            std::string sceneNameOnly = std::regex_replace(kDefaultScene, std::regex(R"(\.fbx$)"), "");
            std::replace(sceneNameOnly.begin(), sceneNameOnly.end(), '/', '_');

            std::string sceneFull(kDefaultScene);
            std::string sceneName = sceneFull.substr(0, sceneFull.find('/')); // e.g. lost_empire
            std::cout << "Scene name: " << sceneName << std::endl;
            int bitRateK = bitRate / 1000;
            newFilePath << "encodedH264/"
                << dateStr << "/"
                << sceneName << "/"
                << sceneNameOnly << "_" << speed << "/"
                << bitRateK << "/"
                << bitRateK << "_" << frameRate << "_" << mHeight << ".h265";

            std::filesystem::create_directories(newFilePath.str().substr(0, newFilePath.str().find_last_of('/')));
            // Copy to buffer safely
            strncpy(szOutFilePath, newFilePath.str().c_str(), sizeof(szOutFilePath));
            szOutFilePath[sizeof(szOutFilePath) - 1] = '\0';
            std::cout << "Output path: " << szOutFilePath << std::endl;


            // // newFilePath << "encodedH264/enc_" << bitRate << "_" << frameRate << "_" <<  mHeight << ".h264";
            // newFilePath << szRefPrefixFilePath << bitRate << "_" << frameRate << "_" <<  mHeight << ".h265";
            // strncpy(szOutFilePath, newFilePath.str().c_str(), sizeof(szOutFilePath));
            // szOutFilePath[sizeof(szOutFilePath) - 1] = '\0'; // Ensure null-termination
            // std::cout << "create szOutFilePath: " << szOutFilePath << std::endl;
            // std::cout << "default scene: " << kDefaultScene << std::endl;

            fpEncOut = new std::ofstream(szOutFilePath, std::ios::out | std::ios::binary | std::ios::app);

            if (!fpEncOut)
            {
                std::ostringstream err;
                err << "/n/n/nunable to open output file: " << szOutFilePath << std::endl;
                throw std::invalid_argument(err.str());
            }
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
    regRsrcInputFence.waitValue = mNInputFenceVal; // GPU will wait for the fence to reach mNInputFenceVal（0） before starting the encoding
    regRsrcInputFence.bWait = true;

    // atomically increment the value of a variable
    // returns the incremented value
    InterlockedIncrement(&mNInputFenceVal);
    regRsrcInputFence.signalValue = mNInputFenceVal; // fence will be set to mNInputFenceVal (1) after the GPU finishes its task.
    regRsrcInputFence.bSignal = true;


    // get directx 12 resource information
    // auto dx12InputTexture = mpRtOut->getNativeHandle().as<ID3D12Resource*>();
    // auto dx12InputTexture = mpRtOut->getNativeHandle().as<ID3D12Resource*>();
    // auto dx12InputTexture = mpRenderGraph->getOutput("FXAA.dst")->asTexture()->getNativeHandle().as<ID3D12Resource*>();
    auto dx12InputTexture = mpRenderGraph->getOutput("TAA.colorOut")->asTexture()->getNativeHandle().as<ID3D12Resource*>();

    D3D12_RESOURCE_DESC desc = dx12InputTexture->GetDesc();
    // Registering the DirectX 12 Resource with NVENC
    // bind raw frame data for GPU-based processing
    // tells NVENC to wait until the fence reaches waitValue before starting to encode,
    // and to signal (update) the fence to signalValue once encoding is complete.
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
    // ID3D12Resource* pTextureRsrc = dx12InputTexture;            // dx12InputTexture is presumably a pointer to a DirectX 12 resource
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
    // cpu waits for the fence to reach the signalValue（1）
    // if pfence < 1, wait
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

// setting up the encoder with the necessary input and output buffers before actual encoding begins.
// It ensures that both input frames (raw video data) and output frames (encoded video data)
void EncodeDecode::mapEncoderResource(uint32_t bufferIndex)
{
    NV_ENC_MAP_INPUT_RESOURCE mapInputResource = {NV_ENC_MAP_INPUT_RESOURCE_VER};
    mapInputResource.registeredResource = mVRegisteredResources[bufferIndex];
    // NVENC_API_CALL is a macro or function that executes the nvEncMapInputResource
    // maps the registered resource for encoding
    NVENC_API_CALL(mNVEnc.nvEncMapInputResource(mHEncoder, &mapInputResource));
    // mappedResource obtained from mapInputResource is stored in mVMappedInputBuffers
    mVMappedInputBuffers[bufferIndex] = mapInputResource.mappedResource;

    // initialize an instance for mapping the output buffer
    NV_ENC_MAP_INPUT_RESOURCE mapInputResourceBitstreamBuffer = {NV_ENC_MAP_INPUT_RESOURCE_VER};
    //  resources where the encoded bitstream will be written
    mapInputResourceBitstreamBuffer.registeredResource = mVRegisteredResourcesOutputBuffer[bufferIndex];
    NVENC_API_CALL(mNVEnc.nvEncMapInputResource(mHEncoder, &mapInputResourceBitstreamBuffer));
    // output buffer is encoded video data
    mVMappedOutputBuffers[bufferIndex] = mapInputResourceBitstreamBuffer.mappedResource;
}

void EncodeDecode::cpuWaitForFencePoint(ID3D12Fence* pFence, uint64_t nFenceValue)
{
    UINT64 val = pFence->GetCompletedValue();
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
        std::cout << "Failed to encode frame.\n";
    }
}

/*
map resources into memory
set picture params
*/
NVENCSTATUS EncodeDecode::encodeFrameBuffer()
{
    NV_ENC_PIC_PARAMS picParams = {};
    // setting up the encoder with input and output buffer
    // input frames is raw video data and output frames is encoded video data
    // assign register resources to mVMappedInputBuffers, mVMappedOutputBuffers
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
    // encoded results written to mVOutputRsrc
    NVENCSTATUS nvStatus = mNVEnc.nvEncEncodePicture(mHEncoder, &picParams);


    //static int countEncode = 0;
    //std::cout << "Encode frame found : " << countEncode++ << "\n\n ";
    static int fCount = 0;

    waitForCompletionEvent(0); // wait for nvEncEncodePicture to finish

    // write encoded frames to out_.h264
    if (outputEncodedFrames)
    {

        {
            // std::cout << "fCount inside: " << fCount++ << std::endl;
            // std::cout << "frameLimit: " << frameLimit << std::endl;
            // std::cout << "write encoded h264 to: " << szOutFilePath << std::endl;
            fpEncOut->write(reinterpret_cast<char*>(mVEncodeOutData.data()), mVEncodeOutData.size());
        }
    }
    // fCount += 1;
    // std::cout << "fCount: " << fCount << std::endl;

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

    // The input resources (mVMappedInputBuffers[0]) used in the encoding process are unmapped
    NVENC_API_CALL(mNVEnc.nvEncUnmapInputResource(mHEncoder, mVMappedInputBuffers[0]));
    mVMappedInputBuffers[0] = nullptr;

    NVENC_API_CALL(mNVEnc.nvEncUnmapInputResource(mHEncoder, mVMappedOutputBuffers[0]));
    mVMappedOutputBuffers[0] = nullptr;

    // std::cout << "write bitstream successfully!\n";

    return nvStatus;
}

void EncodeDecode::makeDefaultDecodingParams(CUVIDDECODECREATEINFO* pInitializeParams)
{
    pInitializeParams->CodecType = cudaVideoCodec_HEVC; // cudaVideoCodec_HEVC= h265, cudaVideoCodec_H264 = h264
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
    videoParserParameters.pfnSequenceCallback = nullptr; // nullptr
    videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc; // called once per frame
    videoParserParameters.pfnDisplayPicture = nullptr;
    videoParserParameters.pfnGetOperatingPoint = nullptr;
    videoParserParameters.pfnGetSEIMsg = nullptr;
    NVDEC_API_CALL(cuvidCreateVideoParser(&mHParser, &videoParserParameters));

    // Check the decoding capabilities
    CUVIDDECODECAPS decodecaps;
    memset(&decodecaps, 0, sizeof(decodecaps));

    decodecaps.eCodecType = cudaVideoCodec_HEVC; // cudaVideoCodec_HEVC, cudaVideoCodec_H264
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

    makeDefaultDecodingParams(&videoDecodeCreateInfo);

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext));
    NVDEC_API_CALL(cuvidCreateDecoder(&mHDecoder, &videoDecodeCreateInfo)); // create decoder
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(nullptr));
    // if (showDecode) {
    presenterPtr = new FramePresenterD3D11(mCudaContext, mWidth, mHeight);
    // }
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
    // if (presenterPtr) {
    presenterPtr->PresentDeviceFrame((uint8_t*)mPDecoderRGBAFrame, mWidth * 4, 0);
    // }

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
    // std::cout << "speed: " << radius * 0.25f << "\n"; // 3.24528
    mpScene->setCameraSpeed(10.f); // radius * 0.25f
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 10;

    // mpCamera->setPosition(Falcor::float3(7.35889, -6.92579, 4.95831));

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

void EncodeDecode::setFrameRate(unsigned int fps)
{
    frameRate = fps; // Assign the private member
    frameLimit = frameRate + numOfFrames * frameRate / 30.0; // 68, 34, 45, 30
    // last_send_time = std::chrono::steady_clock::now();
    std::cout << "setFrameRate  " << frameRate << "\n";
    std::cout << "frameLimit  " << frameLimit << "\n";
    // std::cout << "setvlast_send_time  " << "\n";
}


void EncodeDecode::setSpeed(unsigned int input)
{
    speed = input;
    std::cout << "setSpeed  " << speed << "\n";
}


void EncodeDecode::setRefPrefix(std::string scene, unsigned int speedInput)
{
    std::string fullPath = std::string(szRefPrefixFilePath) + scene + "_" + std::to_string(speedInput) + "/";

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


void EncodeDecode::setDefaultScene(std::string scenePath)
{
    strcpy(kDefaultScene, scenePath.c_str());
    std::cout << "kDefaultScene  " << kDefaultScene << "\n";
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
    // pRenderContext->signal(mpDecodeFence.get(), mNDecodeFenceVal);
    // pRenderContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));
}


// scene
int runMain(int argc, char** argv)
{
    unsigned int bitrate = std::stoi(argv[1]);
    unsigned int framerate = std::stoi(argv[2]);
    unsigned int width = std::stoi(argv[3]);
    unsigned int height = std::stoi(argv[4]);
    std::string scene = argv[5];
    unsigned int speedInput = std::stoi(argv[6]);
    std::string scenePath = argv[7];

    // unsigned int width = 1920; // 1920 1280
    // unsigned int height = 1080; // 1080 720
    // unsigned int bitrate = 8000;
    // unsigned int framerate = 166;
    // std::string scene = "bedroom";
    // unsigned int speedInput = 1;
    // std::string scenePath = "bedroom/path1_seg1.fbx"; // no texture, objects are black

    std::cout << "\n\nframerate runmain  " << framerate << "\n";
    std::cout << "bitrate runmain  " << bitrate << "\n";
    std::cout << "width runmain  " << width << "\n";
    std::cout << "height runmain  " << height << "\n";
    std::cout << "scene " << scene << std::endl;
    std::cout << "speed " << speedInput << std::endl;
    std::cout << "scenePath " << scenePath << std::endl;
    //std::cout << "\n\nbitrate std::stoi(argv[1])  " << std::stoi(argv[1]) << "/n";
    //std::cout << "\n\nnframerate std::stoi(argv[2])  " << std::stoi(argv[2]) << "/n";

    SampleAppConfig config;
    config.windowDesc.title = "EncodeDecode";
    config.windowDesc.resizableWindow = true;
    config.colorFormat = ResourceFormat::BGRA8Unorm;
    config.windowDesc.width = width;
    config.windowDesc.height = height;

    EncodeDecode encodeDecode(config);
    encodeDecode.setBitRate(bitrate * 1000); // 3000 bits per second,  3000 000 bits per second
    encodeDecode.setFrameRate(framerate);
    // encodeDecode.setRefPrefix(scene, speedInput);
    encodeDecode.setDefaultScene(scenePath);
    encodeDecode.setSpeed(speedInput);

    return encodeDecode.run();
}

int main(int argc, char** argv)
{
    std::cout << "Current Path: " << std::filesystem::current_path() << std::endl;
    // unsigned int bitrate = 3000;
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
