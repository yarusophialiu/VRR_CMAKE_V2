// EncodeDecode.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once

#include <filesystem>
#include <iostream>
#include <curl/curl.h>
#include <d3d12.h>
#include <d3d11.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <iomanip>
#include <thread>
#include <random>
#include <chrono>
#include <cppcodec/base64_rfc4648.hpp>

#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/TextRenderer.h"
#include "Core/API/NativeHandleTraits.h"
#include "RenderGraph/RenderGraph.h"

#include "ColorSpace.h"
// #include "NvCodecUtils.h"
#include "EncodeDecode.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvEncodeAPI.h"

#include <Windows.h>
#include <wingdi.h>
#include "FramePresenterD3D11.h"

#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cstdio>
#include <combaseapi.h>

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


double getCurrentTime()
{
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    auto nowInSeconds = duration_cast<duration<double>>(now.time_since_epoch());
    return nowInSeconds.count();
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

// Function to asynchronously send patch and velocity to the HTTP server
// const std::vector<std::vector<float>>& patch
void send_http_request_async(const std::vector<uint8_t>& patchData, float velocity) {
    std::thread([patchData, velocity]() {
        auto start = std::chrono::high_resolution_clock::now();

        CURLM* multi_handle;
        CURL* easy_handle;
        // When still_running becomes 0, it means all HTTP requests have completed, and you can proceed with any post-processing steps or cleanup.
        int still_running = 0; // Number of active transfers

        // Create a JSON object with patch and velocity
        nlohmann::json j;
        // Base64 encode the patch data and add to the JSON
        std::string encodedPatch = base64_encode_patch(patchData);
        j["patch"] = encodedPatch;
        j["velocity"] = velocity;

        // Convert JSON to string
        std::string json_str = j.dump();
        // Initialize curl easy and multi handles
        easy_handle = curl_easy_init();
        multi_handle = curl_multi_init();

        if (easy_handle && multi_handle) {
            // Set URL and POST data
            curl_easy_setopt(easy_handle, CURLOPT_URL, "http://localhost:8000/predict");
            curl_easy_setopt(easy_handle, CURLOPT_POSTFIELDS, json_str.c_str());

            // Set HTTP headers
            struct curl_slist* headers = NULL;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(easy_handle, CURLOPT_HTTPHEADER, headers);
            // Add easy handle to multi handle
            curl_multi_add_handle(multi_handle, easy_handle);

            auto now = std::chrono::system_clock::now();
            std::time_t currentTime = std::chrono::system_clock::to_time_t(now); // getting calendar time
            std::tm* localTime = std::localtime(&currentTime);
            std::cout << "Current local time: " << std::put_time(localTime, "%Y-%m-%d %H:%M:%S") << std::endl;
            auto stop1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds_stop0 = stop1 - start;
            std::cout << "Before send patch takes: " << elapsed_seconds_stop0.count() << " seconds\n";

            // Perform the request asynchronously
            curl_multi_perform(multi_handle, &still_running);

            // Polling loop to check the status of the request
            while (still_running) {
                int numfds;
                curl_multi_wait(multi_handle, NULL, 0, 1000, &numfds);  // Wait for data/events
                curl_multi_perform(multi_handle, &still_running);       // Perform any outstanding transfers
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds_stop1 = end - stop1;
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "Request round trip: " << elapsed_seconds_stop1.count() << " seconds\n";
            std::cout << "Request completed in: " << elapsed_seconds.count() << " seconds\n";

            curl_multi_remove_handle(multi_handle, easy_handle);
            curl_easy_cleanup(easy_handle);
            curl_multi_cleanup(multi_handle);
        }
    }).detach(); // Detach the thread to run independently
}


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
static const Falcor::float4 kClearColor(0.f, 0.f, 0.0f, 1);


// constructor
EncodeDecode::EncodeDecode(const SampleAppConfig& config) : SampleApp(config)
{
    /*1422x800 dec not working, new pairs: 1536, 1200; 864, 676*/
        mConfig = config;
        mWidth = config.windowDesc.width;   // 1920, 4096, 1280, 854, 640, 960, 1024, 1280, 1440, 1423
        mHeight = config.windowDesc.height; // 1080, 2160, 720, 480, 360, 540, 600, 800, 900, 800

        std::cout << '\n';
        std::cout << "mWidth: " << mWidth << std::endl;
        std::cout << "mHeight: " << mHeight << std::endl;
        std::cout << '\n';

        // const ref<Device>& device = getDevice(); // get from falcor, mpDevice = device.get();
        mpDevice = getDevice();
        mpD3D12Device = mpDevice->getNativeHandle().as<ID3D12Device*>();

        mpDecodeFence = mpDevice->createFence();
        mNInputFenceVal = 0;
        mNOutputFenceVal = 0;
        mNDecodeFenceVal = 0;

        mNVEnc = {};
        mEBufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
        mCudaDevice = 0;
        mpRtOut = getDevice()->createTexture2D(mWidth, mHeight, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        std::cout << "bitrate: " << bitRate << std::endl;

        // cast into directx 12 using: ->getNativeHandle().as<ID3D12Resource*>();
        // falcor's device, createtexture3d
        mPDecoderOutputTexture360 = getDevice()->createTexture2D(640, 360, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPDecoderOutputTexture480 = getDevice()->createTexture2D(854, 480, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPDecoderOutputTexture720 = getDevice()->createTexture2D(1280, 720, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPDecoderOutputTexture864 = getDevice()->createTexture2D(1536, 864, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPDecoderOutputTexture1080 = getDevice()->createTexture2D(1920, 1080, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);

        mPEncoderInputTexture360 = getDevice()->createTexture2D(640, 360, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture480 = getDevice()->createTexture2D(854, 480, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture720 = getDevice()->createTexture2D(1280, 720, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture864 = getDevice()->createTexture2D(1536, 864, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture1080 = getDevice()->createTexture2D(1920, 1080, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
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
    mpRenderContextDecode = pRenderContext;

    initEncoder();
    initDecoder();
    std::cout << "load scene: " << std::endl;

    loadScene(kDefaultScene, getTargetFbo().get());

    Properties gBufferProps = {};
    Properties fXAAProps = {};

    // RenderGraph有DirectedGraph, DirectedGraph 存储了 PassId 和 EdgeId 的关系，
    // 而 RenderGraph 则存储了两个 Id 所指向的资源。而这些关系的产生都在 addPass 和 addEdge 中完成
    mpRenderGraph = RenderGraph::create(mpDevice, "EncodeDecode");
    mpRenderGraph->createPass("GBuffer", "GBufferRaster", gBufferProps);
    mpRenderGraph->createPass("TAA", "TAA", fXAAProps);

    mpRenderGraph->onResize(getTargetFbo().get());
    mpRenderGraph->setScene(mpScene);
    mpRenderGraph->addEdge("GBuffer.mvec", "TAA.motionVecs"); // source texture, output texture
    ////mpRenderGraph->markOutput("GBuffer.mvec"); // Mark a render pass output as the graph's output.
    mpRenderGraph->markOutput("TAA.colorOut");
    mpRenderGraph->setInput("TAA.colorIn", mpRtOut);
    mpRenderGraph->compile(pRenderContext, log);

    // allocate memory so encoder can work with what we need
    makeEncoderInputBuffers(6);
    makeEncoderOutputBuffers(1); // 6

    // uint8_t* data = (uint8_t*)malloc(640 * 360 * 4);
    // memset(data, 255, 640 * 360 * 4);
    // pRenderContext->updateTextureData(mPEncoderInputTexture360.get(), data);
}

/*resize window changes the size of presenter of the decoded frame*/
void EncodeDecode::onResize(uint32_t width, uint32_t height)
{
    float h = (float)height; // 1080, 2160, 720,
    float w = (float)width;  // 1920, 3840, 1280,

    // std::cout << '\n';
    // std::cout << "Height: " << h << std::endl;
    // std::cout << '\n';

    if (mpCamera)
    {
        // mpCamera->setPosition(Falcor::float3(0.107229, 1.44531, -1.40544));
        mpCamera->setFocalLength(18);
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }
}

// Function to extract a 128x128 patch from the rendered frame
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

// called in sampleapp renderframe()
void EncodeDecode::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    double startTime = 0.0;
    startTime = getCurrentTime(); // Capture the start time

    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);
    static double timeSecs = 0;

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
            renderRT(mpRenderContextDecode, pTargetFbo, fcount, mWidth, mHeight); // mpRenderContextDecode pRenderContext
        else
            renderRaster(pRenderContext, pTargetFbo);

        mpRenderGraph->execute(mpRenderContextDecode); // mpRenderContextDecode pRenderContext

        // blit from one frame buffer (or texture) to another
        // displaying the final rendered image to the screen, framebuffer represents the final render target, usually the screen or a backbuffer
        // pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), pTargetFbo->getRenderTargetView(0));

        // // send_patch_if_needed();
        // if (fcount == 100) {
        //     ref<Texture> frameTexture = mpRenderGraph->getOutput("TAA.colorOut")->asTexture();
        //     uint32_t frameWidth = frameTexture->getWidth();
        //     uint32_t frameHeight = frameTexture->getHeight();
        //     // auto format = frameTexture->getFormat(); // BGRA8Unorm (47)
        //     uint32_t patchWidth = 128;
        //     uint32_t patchHeight = 128;

        //     std::vector<uint8_t> renderedFrameVal = pRenderContext->readTextureSubresource(frameTexture.get(), 0);
        //     std::vector<uint8_t> patchData(patchWidth * patchHeight * 3);
        //     extract_patch_from_frame(renderedFrameVal, frameWidth, frameHeight, patchWidth, patchHeight, patchData);
        //     stbi_write_png("patch_output_rgba.png", patchWidth, patchHeight, 3, patchData.data(), patchWidth * 3);

        //     send_http_request_async(patchData, 0.75);
        //     // send_http_request_async(0.75);
        // }
        // Sleep(100);
        // std::cout << "sleep here \n";

        cpuWaitForFencePoint(mpDecodeFence->getNativeHandle().as<ID3D12Fence*>(), mNDecodeFenceVal);
        // pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), pTargetFbo->getRenderTargetView(0));

        // blit mprtout into smaller texture, want to encode smaller texture
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture360->getRTV(0)); // mPDecoderOutputTexture1080
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture480->getRTV(0)); // mPDecoderOutputTexture1080
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture720->getRTV(0)); // mPDecoderOutputTexture1080
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture864->getRTV(0)); // mPDecoderOutputTexture1080
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture1080->getRTV(0)); // mPDecoderOutputTexture1080

        if (outputReferenceFrames && (fCount_rt > 0))
        {
            // std::cout<< "fCount_rt-frameRate " << fCount_rt-frameRate << "\n";
            snprintf(szRefOutFilePath, sizeof(szRefOutFilePath), "%s%d.bmp", refBaseFilePath, fCount_rt); // fCount_rt-frameRate
            // mpRtOut->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);
            mpRenderGraph->getOutput("TAA.colorOut")->asTexture()->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);
        }

        encodeFrameBuffer(); // write encoded data into h264
        decodeFrameBuffer(); // mDecodedFrame updated, then call handlePictureDecode
        getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20}); // print fps on the screen


        // std::cout << "frameCounter " << frameCounter << "\n";
        // std::cout << "fcount " << fcount << "\n";
        // frameCounter++;

        if (fCount_rt >= 2)  // 2
        {
            std::cout << "\nfCount_rt: " << fCount_rt << "\n";
            // mWDecodeLock = 0;


            if (mDecodeLock != 0)
            {

                pRenderContext->blit(mPDecoderOutputTexture1080->getSRV(), pTargetFbo->getRenderTargetView(0));

                // if (mHeight == 360)
                // {
                //     // pRenderContext->updateTextureData(mPDecoderOutputTexture360.get(), mDecodedFrame);
                //     //  the decoded texture (retrieved via getSRV()) is copied or rendered to the render target view (retrieved via getRenderTargetView(0) of the framebuffer).
                //     // This is likely part of a process to display the texture onto the screen or an offscreen buffer
                //     std::cout << "onframerenderer mheight = 360" << "\n";
                //     pRenderContext->blit(mPDecoderOutputTexture360->getSRV(), pTargetFbo->getRenderTargetView(0)); // mPDecoderOutputTexture1080
                // } else if (mHeight == 480)
                // {
                //     pRenderContext->blit(mPDecoderOutputTexture480->getSRV(), pTargetFbo->getRenderTargetView(0));
                // } else if (mHeight == 720)
                // {
                //     std::cout << "onframerenderer mheight = 720" << "\n";
                //     pRenderContext->blit(mPDecoderOutputTexture720->getSRV(), pTargetFbo->getRenderTargetView(0));
                // } else if (mHeight == 864)
                // {
                //     pRenderContext->blit(mPDecoderOutputTexture864->getSRV(), pTargetFbo->getRenderTargetView(0));
                // } else
                // {
                //     std::cout << "onframerenderer mheight = 1080" << "\n";
                // }
            }

            double endTime = 0.0;
            endTime = getCurrentTime();

            float elapsedTime = endTime - startTime; // Time since last frame
            // std::cout << "elapsedTime: " << elapsedTime << "\n";

            if (elapsedTime < targetFrameTime)
            {
                // Insert a delay to match the desired FPS
                // std::cout << "start sleepoing, targetFrameTime: " << targetFrameTime << "\n";
                float sleepTime = targetFrameTime - elapsedTime;
                std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
            }

        //    if (outputDecodedFrames)
        //     {
        //         snprintf(szDecOutFilePath, sizeof(szDecOutFilePath), "%s%d.bmp", decBaseFilePath, fcount);
        //         writeBMP(szDecOutFilePath, mPHostRGBAFrame, mWidth, mHeight);
        //     }

            if (mResolutionChange == -1)
            {
                //setFrameRate(10);
                setResolution(640, 360);
                mResolutionChange = 0;
            } else if (mResolutionChange == 1) {

                setResolution(1920, 1080);
                mResolutionChange = 0;
            }

            // if (frameLimit > 0 && fcount >= frameLimit)
            // {
            //     std::exit(0);
            // }
        }
        fCount_rt += 1;
        ++fcount;
        // std::cout << "fcount " << fcount << "\n";
        timeSecs += 1.0 / frameRate;
    }

    mOldWidth = mWidth;
    mOldHeight = mHeight;


}

void EncodeDecode::onGuiRender(Gui* pGui)
{
    // Gui::Window w(pGui, "VRR Settings", {300, 400}, {10, 80});

    // w.checkbox("Ray Trace", mRayTrace);
    // w.checkbox("Use Depth of Field", mUseDOF);
    // if (w.button("Load Scene"))
    // {
    //     std::filesystem::path path;
    //     if (openFileDialog(Scene::getFileExtensionFilters(), path))
    //     {
    //         loadScene(path, getTargetFbo().get());
    //     }
    // }

    // mpScene->renderUI(w);
}

bool EncodeDecode::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.key == Falcor::Input::Key::Up) {

        mResolutionChange = 1;

        return true;
    } else if (keyEvent.key == Falcor::Input::Key::Down) {

        mResolutionChange = -1;

        return true;
    }

    return false;
}

bool EncodeDecode::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene && mpScene->onMouseEvent(mouseEvent);
}

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
    pIntializeParams->maxEncodeWidth = 1920; // mWidth
    pIntializeParams->maxEncodeHeight = 1080;
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
        // pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 =
        //     (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 2 : 0;
        if (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
        {
            pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
        }
        pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = pIntializeParams->encodeConfig->gopLength;
    }
    else if (pIntializeParams->encodeGUID == NV_ENC_CODEC_AV1_GUID)
    {
        // pIntializeParams->encodeConfig->encodeCodecConfig.av1Config.pixelBitDepthMinus8 =
        //     (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT) ? 2 : 0;
        // pIntializeParams->encodeConfig->encodeCodecConfig.av1Config.inputPixelBitDepthMinus8 =
        //     (mEBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT) ? 2 : 0;
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
    // std::cout << "\nmaxBitRate " << mEncodeConfig.rcParams.maxBitRate << "\n";

    NVENC_API_CALL(mNVEnc.nvEncInitializeEncoder(mHEncoder, &mEncoderInitializeParams));

    mNEncoderBuffer = mEncodeConfig.frameIntervalP + mEncodeConfig.rcParams.lookaheadDepth;

    mEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    mVPCompletionEvent.resize(6, nullptr); // TODO: 1 event should be fine too

    for (uint32_t i = 0; i < mVPCompletionEvent.size(); i++)
    {
        mVPCompletionEvent[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
    }


    if (outputEncodedFrames)
        {
            std::ostringstream newFilePath;
            // newFilePath << "encodedH264/enc_" << bitRate << "_" << frameRate << "_" <<  mHeight << ".h264";
            newFilePath << szRefPrefixFilePath << bitRate << "_" << frameRate << "_" <<  mHeight << ".h264";

            strncpy(szOutFilePath, newFilePath.str().c_str(), sizeof(szOutFilePath));
            szOutFilePath[sizeof(szOutFilePath) - 1] = '\0'; // Ensure null-termination
            std::cout << "create szOutFilePath: " << szOutFilePath << std::endl;
            std::cout << "default scene: " << kDefaultScene << std::endl;

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
    registerResource.width = width; // width;
    registerResource.height = height; // height;
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

int EncodeDecode::getEncoderFrameSize(int width, int height)
{
    switch (mEBufferFormat)
    {
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
    case NV_ENC_BUFFER_FORMAT_NV12:
        return width * (height + (height + 1) / 2);
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
        return 2 * width * (height + (height + 1) / 2);
    case NV_ENC_BUFFER_FORMAT_YUV444:
        return width * height * 3;
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
        return 2 * width * height * 3;
    case NV_ENC_BUFFER_FORMAT_ARGB:
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    case NV_ENC_BUFFER_FORMAT_AYUV:
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ABGR10:
        return 4 * width * height;
    default:
        return 0;
    }
}

uint32_t EncodeDecode::getEncoderOutputBufferSize(int width, int height)
{
    uint32_t bufferSize = getEncoderFrameSize(width, height) * 4;
    bufferSize = ALIGN_UP(bufferSize, 4);
    return bufferSize;
}

D3D12_RESOURCE_DESC EncodeDecode::createEncoderResourceDesc(int width, int height)
{
    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = width; // mWidth; 1920
    resourceDesc.Height = height; // mHeight; 1080
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    resourceDesc.Format = GetD3D12Format(mEBufferFormat);

    return resourceDesc;
}

void EncodeDecode::makeEncoderInputBuffers(int32_t numInputBuffers)
{
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    // TODO: remove 1 resource
    D3D12_RESOURCE_DESC resourceDesc = createEncoderResourceDesc(mWidth, mHeight);
    D3D12_RESOURCE_DESC resourceDesc360 = createEncoderResourceDesc(640, 360);
    D3D12_RESOURCE_DESC resourceDesc480 = createEncoderResourceDesc(854, 480);
    D3D12_RESOURCE_DESC resourceDesc720 = createEncoderResourceDesc(1280, 720);
    D3D12_RESOURCE_DESC resourceDesc864 = createEncoderResourceDesc(1536, 864);
    D3D12_RESOURCE_DESC resourceDesc1080 = createEncoderResourceDesc(1920, 1080);

    mVInputBuffers.resize(numInputBuffers); // necessary, otherwise throw vector out of range error
    mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mVInputBuffers[0]));
    mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc360, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mVInputBuffers[1]));
    mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc480, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mVInputBuffers[2]));
    mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc720, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mVInputBuffers[3]));
    mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc864, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mVInputBuffers[4]));
    mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc1080, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mVInputBuffers[5]));

    registerEncoderInputResources(mWidth, mHeight, mEBufferFormat);

    for (int i = 0; i < numInputBuffers; i++)
    {
        NV_ENC_INPUT_RESOURCE_D3D12* pInpRsrc = new NV_ENC_INPUT_RESOURCE_D3D12();
        memset(pInpRsrc, 0, sizeof(NV_ENC_INPUT_RESOURCE_D3D12));
        pInpRsrc->inputFencePoint.pFence = mpInputFence;

        mVInputRsrc.push_back(pInpRsrc);
    }
}

D3D12_RESOURCE_DESC EncodeDecode::createEncoderOutputResourceDesc(int width, int height)
{
    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = getEncoderOutputBufferSize(width, height);
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    return resourceDesc;
}

void EncodeDecode::makeEncoderOutputBuffers(uint32_t numOutputBuffers)
{
    HRESULT hr = S_OK;

    D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_COPY_DEST;

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    // TODO: resourceDesc might be unecessary
    D3D12_RESOURCE_DESC resourceDesc = createEncoderOutputResourceDesc(1920, 1080); // (mWidth, mHeight);
    // D3D12_RESOURCE_DESC resourceDesc360 = createEncoderOutputResourceDesc(640, 360);
    // D3D12_RESOURCE_DESC resourceDesc480 = createEncoderOutputResourceDesc(854, 480);
    // D3D12_RESOURCE_DESC resourceDesc720 = createEncoderOutputResourceDesc(1280, 720);
    // D3D12_RESOURCE_DESC resourceDesc864 = createEncoderOutputResourceDesc(1536, 864);
    // D3D12_RESOURCE_DESC resourceDesc1080 = createEncoderOutputResourceDesc(1920, 1080);

    mVOutputBuffers.resize(numOutputBuffers);
    mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, initialResourceState, nullptr, IID_PPV_ARGS(&mVOutputBuffers[0]));
    // mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc360, initialResourceState, nullptr, IID_PPV_ARGS(&mVOutputBuffers[1]));
    // mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc480, initialResourceState, nullptr, IID_PPV_ARGS(&mVOutputBuffers[2]));
    // mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc720, initialResourceState, nullptr, IID_PPV_ARGS(&mVOutputBuffers[3]));
    // mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc864, initialResourceState, nullptr, IID_PPV_ARGS(&mVOutputBuffers[4]));
    // mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc1080, initialResourceState, nullptr, IID_PPV_ARGS(&mVOutputBuffers[5]));

    registerEncoderOutputResources();

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


    // dx12InputTexture is presumably a pointer to a DirectX 12 resource
    // auto dx12InputTexture = mpRtOut->getNativeHandle().as<ID3D12Resource*>();
    auto dx12InputTexture = mpRenderGraph->getOutput("TAA.colorOut")->asTexture()->getNativeHandle().as<ID3D12Resource*>();
    // input texture should be of different resolution
    auto dx12InputTexture360 = mPEncoderInputTexture360->getNativeHandle().as<ID3D12Resource*>();
    auto dx12InputTexture480 = mPEncoderInputTexture480->getNativeHandle().as<ID3D12Resource*>();
    auto dx12InputTexture720 = mPEncoderInputTexture720->getNativeHandle().as<ID3D12Resource*>();
    auto dx12InputTexture864 = mPEncoderInputTexture864->getNativeHandle().as<ID3D12Resource*>();
    auto dx12InputTexture1080 = mPEncoderInputTexture1080->getNativeHandle().as<ID3D12Resource*>();


    D3D12_RESOURCE_DESC desc = dx12InputTexture->GetDesc();
    D3D12_RESOURCE_DESC desc360 = dx12InputTexture360->GetDesc();
    D3D12_RESOURCE_DESC desc480 = dx12InputTexture480->GetDesc();
    D3D12_RESOURCE_DESC desc720 = dx12InputTexture720->GetDesc();
    D3D12_RESOURCE_DESC desc864 = dx12InputTexture864->GetDesc();
    D3D12_RESOURCE_DESC desc1080 = dx12InputTexture1080->GetDesc();
    // Registering the DirectX 12 Resource with NVENC
    // bind raw frame data for GPU-based processing
    // tells NVENC to wait until the fence reaches waitValue before starting to encode,
    // and to signal (update) the fence to signalValue once encoding is complete.
    NV_ENC_REGISTERED_PTR registeredPtr = registerNVEncResource(
        dx12InputTexture,                   // DirectX 12 resource to be registered
        NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, // Type of the input resource (DirectX texture in this case)
        width, height, 0,                   // 0 Pitch (row pitch of the resource), 0 for automatic calculation
        bufferFormat,                       // Format of the input buffer (e.g., DXGI_FORMAT_R8G8B8A8_UNORM)
        NV_ENC_INPUT_IMAGE,                 // Type of input image (e.g., for image input)
        &regRsrcInputFence                  // Pointer to an NV_ENC_FENCE_POINT_D3D12 structure
    );
    NV_ENC_REGISTERED_PTR registeredPtr360 = registerNVEncResource(dx12InputTexture360, NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, 640, 360, 0, bufferFormat, NV_ENC_INPUT_IMAGE, &regRsrcInputFence);
    NV_ENC_REGISTERED_PTR registeredPtr480 = registerNVEncResource(dx12InputTexture480, NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, 854, 480, 0, bufferFormat, NV_ENC_INPUT_IMAGE, &regRsrcInputFence);
    NV_ENC_REGISTERED_PTR registeredPtr720 = registerNVEncResource(dx12InputTexture720, NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, 1280, 720, 0, bufferFormat, NV_ENC_INPUT_IMAGE, &regRsrcInputFence);
    NV_ENC_REGISTERED_PTR registeredPtr864 = registerNVEncResource(dx12InputTexture864, NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, 1536, 864, 0, bufferFormat, NV_ENC_INPUT_IMAGE, &regRsrcInputFence);
    NV_ENC_REGISTERED_PTR registeredPtr1080 = registerNVEncResource(dx12InputTexture1080, NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, 1920, 1080, 0, bufferFormat, NV_ENC_INPUT_IMAGE, &regRsrcInputFence);

    // Creating an NvEncInputFrame, preparing data structures for use in the NVENC encoding process
    NvEncInputFrame inputframe = {};                            // {}: initialize all its members set to their default value
    NvEncInputFrame inputframe360 = {};
    NvEncInputFrame inputframe480 = {};
    NvEncInputFrame inputframe720 = {};
    NvEncInputFrame inputframe864 = {};
    NvEncInputFrame inputframe1080 = {};
    // describe two separate subresources (e.g., two mip levels or two array slices) within a texture or buffer that you want to copy or upload
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint[2]; // Assuming each resource has 2 subresources
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint360[2]; // Assuming each resource has 2 subresources
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint480[2]; // Assuming each resource has 2 subresources
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint720[2]; // Assuming each resource has 2 subresources
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint864[2];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint1080[2];
    // Gets a resource layout that can be copied. Helps the app fill-in D3D12_PLACED_SUBRESOURCE_FOOTPRINT and
    // D3D12_SUBRESOURCE_FOOTPRINT when suballocating space in upload heaps.
    mpD3D12Device->GetCopyableFootprints(&desc, 0, 1, 0, inputUploadFootprint, nullptr, nullptr, nullptr);
    mpD3D12Device->GetCopyableFootprints(&desc360, 0, 1, 0, inputUploadFootprint360, nullptr, nullptr, nullptr);
    mpD3D12Device->GetCopyableFootprints(&desc480, 0, 1, 0, inputUploadFootprint480, nullptr, nullptr, nullptr);
    mpD3D12Device->GetCopyableFootprints(&desc720, 0, 1, 0, inputUploadFootprint720, nullptr, nullptr, nullptr);
    mpD3D12Device->GetCopyableFootprints(&desc864, 0, 1, 0, inputUploadFootprint864, nullptr, nullptr, nullptr);
    mpD3D12Device->GetCopyableFootprints(&desc1080, 0, 1, 0, inputUploadFootprint1080, nullptr, nullptr, nullptr);

    inputframe.inputPtr = (void*)dx12InputTexture;
    inputframe.numChromaPlanes = getEncoderNumChromaPlanes(bufferFormat);
    inputframe.bufferFormat = bufferFormat;
    inputframe.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    inputframe.pitch = inputUploadFootprint[0].Footprint.RowPitch;

    inputframe360.inputPtr = (void*)dx12InputTexture360;
    inputframe360.numChromaPlanes = getEncoderNumChromaPlanes(bufferFormat);
    inputframe360.bufferFormat = bufferFormat;
    inputframe360.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    inputframe360.pitch = inputUploadFootprint360[0].Footprint.RowPitch;

    inputframe480.inputPtr = (void*)dx12InputTexture480;
    inputframe480.numChromaPlanes = getEncoderNumChromaPlanes(bufferFormat);
    inputframe480.bufferFormat = bufferFormat;
    inputframe480.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    inputframe480.pitch = inputUploadFootprint480[0].Footprint.RowPitch;

    inputframe720.inputPtr = (void*)dx12InputTexture720;
    inputframe720.numChromaPlanes = getEncoderNumChromaPlanes(bufferFormat);
    inputframe720.bufferFormat = bufferFormat;
    inputframe720.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    inputframe720.pitch = inputUploadFootprint720[0].Footprint.RowPitch;

    inputframe864.inputPtr = (void*)dx12InputTexture864;
    inputframe864.numChromaPlanes = getEncoderNumChromaPlanes(bufferFormat);
    inputframe864.bufferFormat = bufferFormat;
    inputframe864.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    inputframe864.pitch = inputUploadFootprint864[0].Footprint.RowPitch;

    inputframe1080.inputPtr = (void*)dx12InputTexture1080;
    inputframe1080.numChromaPlanes = getEncoderNumChromaPlanes(bufferFormat);
    inputframe1080.bufferFormat = bufferFormat;
    inputframe1080.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    inputframe1080.pitch = inputUploadFootprint1080[0].Footprint.RowPitch;

    mVRegisteredResources.push_back(registeredPtr);
    mVInputFrames.push_back(inputframe);

    mVRegisteredResources.push_back(registeredPtr360);
    mVInputFrames.push_back(inputframe360);

    mVRegisteredResources.push_back(registeredPtr480);
    mVInputFrames.push_back(inputframe480);

    mVRegisteredResources.push_back(registeredPtr720);
    mVInputFrames.push_back(inputframe720);

    mVRegisteredResources.push_back(registeredPtr864);
    mVInputFrames.push_back(inputframe864);

    mVRegisteredResources.push_back(registeredPtr1080);
    mVInputFrames.push_back(inputframe1080);

    mVMappedInputBuffers.resize(6); // don't need

    // CPU wait for register resource to finish, waits for the fence to reach the signalValue（1）
    // if pfence < 1, wait
    cpuWaitForFencePoint((ID3D12Fence*)regRsrcInputFence.pFence, regRsrcInputFence.signalValue);
    //}
}

void EncodeDecode::registerEncoderOutputResources()
{
    // TODO: only need 1 outputbuffer, set to 1080p
    // bfrSize = getEncoderOutputBufferSize(mWidth, mHeight)
    NV_ENC_REGISTERED_PTR registeredPtr = registerNVEncResource(mVOutputBuffers[0], NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, \
                getEncoderOutputBufferSize(1920, 1080), 1, 0, NV_ENC_BUFFER_FORMAT_U8, NV_ENC_OUTPUT_BITSTREAM); // mWidth, mHeight
    // NV_ENC_REGISTERED_PTR registeredPtr360 = registerNVEncResource(mVOutputBuffers[1], NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, getEncoderOutputBufferSize(640, 360), 1, 0, NV_ENC_BUFFER_FORMAT_U8, NV_ENC_OUTPUT_BITSTREAM);
    // NV_ENC_REGISTERED_PTR registeredPtr480 = registerNVEncResource(mVOutputBuffers[2], NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, getEncoderOutputBufferSize(854, 480), 1, 0, NV_ENC_BUFFER_FORMAT_U8, NV_ENC_OUTPUT_BITSTREAM);
    // NV_ENC_REGISTERED_PTR registeredPtr720 = registerNVEncResource(mVOutputBuffers[3], NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, getEncoderOutputBufferSize(1280, 720), 1, 0, NV_ENC_BUFFER_FORMAT_U8, NV_ENC_OUTPUT_BITSTREAM);
    // NV_ENC_REGISTERED_PTR registeredPtr864 = registerNVEncResource(mVOutputBuffers[4], NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, getEncoderOutputBufferSize(1536, 864), 1, 0, NV_ENC_BUFFER_FORMAT_U8, NV_ENC_OUTPUT_BITSTREAM);
    // NV_ENC_REGISTERED_PTR registeredPtr1080 = registerNVEncResource(mVOutputBuffers[5], NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, getEncoderOutputBufferSize(1920, 1080), 1, 0, NV_ENC_BUFFER_FORMAT_U8, NV_ENC_OUTPUT_BITSTREAM);

    mVRegisteredResourcesOutputBuffer.push_back(registeredPtr);
    // mVRegisteredResourcesOutputBuffer.push_back(registeredPtr360);
    // mVRegisteredResourcesOutputBuffer.push_back(registeredPtr480);
    // mVRegisteredResourcesOutputBuffer.push_back(registeredPtr720);
    // mVRegisteredResourcesOutputBuffer.push_back(registeredPtr864);
    // mVRegisteredResourcesOutputBuffer.push_back(registeredPtr1080);

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
    mapInputResourceBitstreamBuffer.registeredResource = mVRegisteredResourcesOutputBuffer[0]; // [bufferIndex];
    NVENC_API_CALL(mNVEnc.nvEncMapInputResource(mHEncoder, &mapInputResourceBitstreamBuffer));
    // output buffer is encoded video data
    mVMappedOutputBuffers[0] = mapInputResourceBitstreamBuffer.mappedResource;
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

int EncodeDecode::getResolutionIndex(int resolution) {
    switch (resolution) {
        case 360:  return 1;
        case 480:  return 2;
        case 720:  return 3;
        case 864:  return 4;
        case 1080: return 5;
        default:   return 0; // Return -1 for unsupported resolutions
    }
}

/*
map resources into memory, set picture params
*/
NVENCSTATUS EncodeDecode::encodeFrameBuffer()
{
    int bufferIndex = getResolutionIndex(mOldHeight);
    NV_ENC_PIC_PARAMS picParams = {};
    // setting up the encoder with input and output buffer
    // input frames is raw video data and output frames is encoded video data
    // assign register resources to mVMappedInputBuffers, mVMappedOutputBuffers
    mapEncoderResource(bufferIndex);

    InterlockedIncrement(&mNOutputFenceVal);

    NV_ENC_INPUT_PTR inputBuffer = mVMappedInputBuffers[bufferIndex];
    NV_ENC_OUTPUT_PTR outputBuffer = mVMappedOutputBuffers[0];

    mVInputRsrc[bufferIndex]->pInputBuffer = inputBuffer;
    mVInputRsrc[bufferIndex]->inputFencePoint.waitValue = mNInputFenceVal;
    mVInputRsrc[bufferIndex]->inputFencePoint.bWait = true;

    mVOutputRsrc[0]->pOutputBuffer = outputBuffer;
    mVOutputRsrc[0]->outputFencePoint.signalValue = mNOutputFenceVal;
    mVOutputRsrc[0]->outputFencePoint.bSignal = true;

    picParams.version = NV_ENC_PIC_PARAMS_VER;
    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picParams.inputBuffer = mVInputRsrc[bufferIndex];
    picParams.bufferFmt = mEBufferFormat;
    picParams.inputWidth = mOldWidth;
    picParams.inputHeight = mOldHeight;
    picParams.outputBitstream = mVOutputRsrc[0];
    picParams.completionEvent = mVPCompletionEvent[bufferIndex];
    // encoded results written to mVOutputRsrc
    NVENCSTATUS nvStatus = mNVEnc.nvEncEncodePicture(mHEncoder, &picParams);

    std::cout << "Endcoding dimensions: " << mWidth << "x" << mHeight << "\n";

    static int fCount = 0;
    waitForCompletionEvent(bufferIndex); // wait for nvEncEncodePicture to finish

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

    // clear the previous encoded frame
    mVEncodeOutData.clear();

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
    NVENC_API_CALL(mNVEnc.nvEncUnmapInputResource(mHEncoder, mVMappedInputBuffers[bufferIndex]));
    mVMappedInputBuffers[bufferIndex] = nullptr;

    NVENC_API_CALL(mNVEnc.nvEncUnmapInputResource(mHEncoder, mVMappedOutputBuffers[0]));
    mVMappedOutputBuffers[0] = nullptr;
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
    pInitializeParams->ulMaxWidth = 1920;// mWidth;
    pInitializeParams->ulMaxHeight = 1080;// mHeight;
    pInitializeParams->ulTargetWidth = mWidth;
    pInitializeParams->ulTargetHeight = mHeight;
    pInitializeParams->enableHistogram = 0;
}


void EncodeDecode::initDecoder()
{
    // Initialise CUDA first, check how many gpus we have, use the first one
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
    CUresult contextError = cuCtxCreate(&mCudaContext, CU_CTX_SCHED_BLOCKING_SYNC, mCudaDevice);
    NVDEC_API_CALL(cuvidCtxLockCreate(&mCtxLock, mCudaContext));
    cuStreamCreate(&mCuvidStream, CU_STREAM_DEFAULT);

    /*
    a parser is a component responsible for analyzing the bitstream or input data and extracting relevant information or structures
    parse the bitstream and perform decoding
    */
    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = cudaVideoCodec_H264; // cudaVideoCodec_HEVC, cudaVideoCodec_H264
    videoParserParameters.ulMaxNumDecodeSurfaces = 4; // >3 works, number of surfaces (decoded frames) in parser’s DPB (decode picture
                                                      // buffer)
    videoParserParameters.ulClockRate = 0;
    videoParserParameters.ulMaxDisplayDelay = 0; // 0 = no delay
    videoParserParameters.pUserData = this;
    videoParserParameters.pfnSequenceCallback = HandleSequenceChangeProc; // nullptr
    videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc; // called once per frame
    videoParserParameters.pfnDisplayPicture = nullptr;
    videoParserParameters.pfnGetOperatingPoint = nullptr;
    videoParserParameters.pfnGetSEIMsg = nullptr;
    NVDEC_API_CALL(cuvidCreateVideoParser(&mHParser, &videoParserParameters));

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
    makeDefaultDecodingParams(&videoDecodeCreateInfo);

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext));
    NVDEC_API_CALL(cuvidCreateDecoder(&mHDecoder, &videoDecodeCreateInfo)); // create decoder
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(nullptr));
    // if (showDecode) {
    // presenterPtr = new FramePresenterD3D11(mCudaContext, mWidth, mHeight);
    // }
    makeDecoderOutputBuffers(); // allocate cuda memory
}

void EncodeDecode::makeDecoderOutputBuffers()
{
    CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr*)&mPDecoderFrame, getDecoderFrameSize()));
    CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr*)&mPDecoderRGBAFrame, 1920 * 1080 * 4));
    // mPHostRGBAFrame = new uint8_t[mWidth * mHeight * 4];
    mPHostRGBAFrame = new uint8_t[1920 * 1080 * 4];
}

/*
called by us once per frame buffer
perform parsing
trigger handlePicutreDecode to perform decoding
*/
void EncodeDecode::decodeFrameBuffer()
{
    // static int bitstream_size = 0;
    CUVIDSOURCEDATAPACKET packet = {0};
    packet.payload = mVEncodeOutData.data();
    packet.payload_size = mVEncodeOutData.size();

    // packet.flags |= CUVID_PKT_DISCONTINUITY;
    // bitstream_size += mVEncodeOutData.size();
    // std::cout << "mVEncodeOutData.size() " << mVEncodeOutData.size() << "\n";
    NVDEC_API_CALL(cuvidParseVideoData(mHParser, &packet));
}

int EncodeDecode::handleSequenceChange(CUVIDEOFORMAT* pFormat) {

    std::cout << "Resolution changed\n";

    unsigned int width = pFormat->coded_width;
    unsigned int height = pFormat->coded_height;

    if (mHDecoder != nullptr)
        {
            std::cout << "setting resolution\n";

            CUVIDRECONFIGUREDECODERINFO decoder_reconifg_info = {0};
            decoder_reconifg_info.ulWidth = mWidth; // mWidth throws cuvidDecodePicture(mHDecoder, pPicParams) returned error 1
            decoder_reconifg_info.ulHeight = mHeight; // mHeight throws cuvidDecodePicture(mHDecoder, pPicParams) returned error 1
            // Post-Processed Output Resolution, ensures that the 640x360 frame is scaled up correctly to 1920x1080
            // scale the decoded scene
            decoder_reconifg_info.ulTargetWidth = 1920; // 1920; if set to mWidth, animation of 640x360 runs on the top left corner
            decoder_reconifg_info.ulTargetHeight = 1080; // 1080;

            // source cropping, ensure that the entire decoded frame is displayed
            decoder_reconifg_info.display_area.left = 0;
            decoder_reconifg_info.display_area.right = decoder_reconifg_info.ulWidth; // mWidth; 1920 左上角出现画中画
            decoder_reconifg_info.display_area.top = 0;
            decoder_reconifg_info.display_area.bottom = decoder_reconifg_info.ulHeight; //mHeight;

            // Aspect Ratio Conversion， ensures that the 640x360 decoded frame is upscaled and displayed within the 1920x1080 target area
            decoder_reconifg_info.target_rect.left = 0;
            decoder_reconifg_info.target_rect.right = 1920; // 1920;
            decoder_reconifg_info.target_rect.top = 0;
            decoder_reconifg_info.target_rect.bottom = 1080; // 1080;

            cuvidReconfigureDecoder(mHDecoder, &decoder_reconifg_info);
        }


    return 1;
}

// callback function: called as soon as picture get decoded
// don't know when it's decoded
// pPicParams provided by the parser, which is initiated in initdecoder
int EncodeDecode::handlePictureDecode(CUVIDPICPARAMS* pPicParams)
{
    static int count = 0;
    // We have parsed an entire frame! Now let's decode it
    // std::cout << "Frame found: " << count++ << "\n\n";

    std::cout << "Pic params width: " << pPicParams->PicWidthInMbs * 16 << "\n";
    std::cout << "Old width: " << mOldWidth << "\n";

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
    // decode surface index (nPicIdx) as input and maps it to one
    // of available output surfaces, post-processes the decoded frame and copy to output surface and
    // returns CUDA device pointer and associated pitch of the output surfaces
    // call cuvidMapVideoFrame() to get the CUDA device pointer and pitch of the output surface that holds the decoded and post-processed frame
    // After the function call, dpSrcFrame will contain the CUDA device pointer to the decoded frame on the GPU.
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
    m.dstPitch = 1920; // 1920; mWidth
    m.WidthInBytes = 1920; // 1920; mWidth
    m.Height = 1080; // 1080; mHeight
    // cuMemcpy2DAsync(&m, mCuvidStream)
    CUDA_DRVAPI_CALL(cuMemcpy2D(&m)); // CUDA function for asynchronously copying memory between two 2D memory regions
    m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((1080 + 1) & ~1)); // updated with a new source device pointer
    m.dstDevice = (CUdeviceptr)(m.dstHost = mPDecoderFrame + m.dstPitch * m.Height);
    m.Height = (int)(ceil(m.Height * 0.5));

    CUDA_DRVAPI_CALL(cuMemcpy2D(&m));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    // cuvidUnmapVideoFrame() must be
    // called to make the output surface available for storing other decoded and post-processed frames
    NVDEC_API_CALL(cuvidUnmapVideoFrame(mHDecoder, dpSrcFrame));

    // color format conversion, convert video frame data from the NV12 format to the BGRA32 format
    // NV12 is a commonly used video format where the Y (luminance) and UV (chrominance) components are stored in separate planes.
    // mPDecoderFrame: pointer to input NV12 frame data
    // mPDecoderRGBAFrame: Pointer to the output buffer where the converted BGRA32 data will be stored,
    // data type is cast to uint8_t*, indicating that the output data is treated as an array of bytes
    // mPDecoderRGBAFrame would represent the frame in BGRA32 format
    // mWidth * 4: The pitch (stride) of the output data
    Nv12ToColor32<BGRA32>(mPDecoderFrame, 1920, (uint8_t*)mPDecoderRGBAFrame, 1920 * 4, 1920, 1080);

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext)); // push the CUDA context (mCudaContext) onto the current thread's CUDA context stack.
    //  copying data from mPDecoderRGBAFrame (GPU device memory) to mPHostRGBAFrame (CPU host memory)
    CUDA_DRVAPI_CALL(cuMemcpyDtoH(mPHostRGBAFrame, mPDecoderRGBAFrame, 1920 * 1080 * 4));

    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL)); // o pop the current CUDA context off the CUDA context stack to release resources

    // show mPDecoderRGBAFrame , access cuda memory from directx12 is difficult
    // create directx12 texture, register with cuda xxx, write to this texture
    // don't need to transfer from cuda to cpu and to directx12
    // do: make the memory using directx12, write to the resource

    // FramePresenterD3D11 presenter(mCudaContext, mWidth, /*mHeight);
    // if (presenterPtr) {
    // pRenderContext->updateTextureData(mDecodedFrame, mPHostRGBAFrame);
    mDecodedFrame = mPHostRGBAFrame;

    // static int fcount = 0;
    // snprintf(szDecOutFilePath, sizeof(szDecOutFilePath), "%s%d.bmp", decBaseFilePath, fcount);
    // writeBMP(szDecOutFilePath, mDecodedFrame, 1920, 1080);
    // ++fcount;

    // presenterPtr->PresentDeviceFrame((uint8_t*)mPDecoderRGBAFrame, mWidth * 4, 0);
    // }
    mpRenderContextDecode->updateTextureData(mPDecoderOutputTexture1080.get(), mDecodedFrame);

    mDecodeLock = 1;
    return 0;
}

int EncodeDecode::getDecoderFrameSize()
{
    float chromaHeightFactor = 0.5;
    int lumaHeight = 1080;// mHeight;
    int chromaHeight = (int)(ceil(lumaHeight * chromaHeightFactor));
    int nChromaPlanes = 1;
    int nBPP = 1;
    // int result = mWidth * (lumaHeight + (chromaHeight * nChromaPlanes)) * nBPP;
    int result = 1920 * (lumaHeight + (chromaHeight * nChromaPlanes)) * nBPP;

    return result;
}


void EncodeDecode::initDirectML()
{
    // // mpD3D12Device
    // DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
	// // DMLCreateDevice(mpD3D12Device, dmlCreateDeviceFlags, 0, &mpDmlDevice);
    // DMLCreateDevice(
    //     mpD3D12Device,
    //     dmlCreateDeviceFlags,
    //     IID_PPV_ARGS(mpDmlDevice->GetAddressOf()));
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

    // Get shader modules and type conformances for types used by the scene.
    // These need to be set on the program in order to use Falcor's material system.
    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    // Get scene defines. These need to be set on any program using the scene.
    auto defines = mpScene->getSceneDefines();

    // This utility wraps the creation of the program and vars, and sets the necessary scene defines.
    ProgramDesc rasterProgDesc;
    rasterProgDesc.addShaderModules(shaderModules);
    rasterProgDesc.addShaderLibrary("Samples/EncodeDecode/EncodeDecode.3d.slang").vsEntry("vsMain").psEntry("psMain");
    rasterProgDesc.addTypeConformances(typeConformances);
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
    targetFrameTime = 1.0f / frameRate;
    // last_send_time = std::chrono::steady_clock::now();
    std::cout << "setFrameRate  " << frameRate << "\n";
    std::cout << "frameLimit  " << frameLimit << "\n";
    std::cout << "targetFrameTime  " << targetFrameTime << "\n";
    // std::cout << "setvlast_send_time  " << "\n";

    if (mHEncoder != nullptr)
    {
        mEncoderInitializeParams.frameRateNum = fps;
        mEncoderInitializeParams.frameRateDen = 1;
        NV_ENC_RECONFIGURE_PARAMS reconfig_params = {0};
        reconfig_params.reInitEncodeParams = mEncoderInitializeParams;
        reconfig_params.resetEncoder = 1;
        // reconfig_params.forceIDR = 1;
        reconfig_params.version = NV_ENC_RECONFIGURE_PARAMS_VER;
        // NvEncReconfigureEncoder(mHEncoder, &reconfig_params);
        NVENC_API_CALL(mNVEnc.nvEncReconfigureEncoder(mHEncoder, &reconfig_params));
    }
}


void EncodeDecode::setResolution(unsigned int width, unsigned int height)
{
    mWidth = width;
    mHeight = height;

    if (mHEncoder != nullptr)
    {
        mEncoderInitializeParams.encodeWidth = width; // width only shows the cropped scene, 1920 works for 360p
        mEncoderInitializeParams.encodeHeight = height; // height;
        mEncoderInitializeParams.darWidth = width;
        mEncoderInitializeParams.darHeight = height;

        NV_ENC_RECONFIGURE_PARAMS reconfig_params;
        // memset(&reconfig_params, 0, sizeof(reconfig_params));
        reconfig_params.reInitEncodeParams = mEncoderInitializeParams;
        reconfig_params.resetEncoder = 0;
        reconfig_params.forceIDR = 0;
        reconfig_params.version = NV_ENC_RECONFIGURE_PARAMS_VER;
        NVENC_API_CALL(mNVEnc.nvEncReconfigureEncoder(mHEncoder, &reconfig_params));

        // if (presenterPtr != nullptr)
        // {
        //     presenterPtr->nWidth = width;
        //     presenterPtr->nHeight = height;
        // }

    }


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


void EncodeDecode::renderRT(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo, int fCount, uint32_t width, uint32_t height)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "renderRT");
    setPerFrameVars(pTargetFbo.get());

    // comment below line to fix artifacts of decoded frames
    // pRenderContext->clearUAV(mpRtOut->getUAV().get(), kClearColor); // kClearColorb is green
    mpScene->raytrace(pRenderContext, mpRaytraceProgram.get(), mpRtVars, Falcor::uint3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1));
    // mpScene->raytrace(pRenderContext, mpRaytraceProgram.get(), mpRtVars, Falcor::uint3(width, height, 1));
    /*
    *
    performing a blit operation from the Shader Resource View (mpRtOut->getSRV()) to the render target view (pTargetFbo->getRenderTargetView(0)) using (pRenderContext).

    blit：block transfer, in graphics means copy pixel data from A to B
    render target: destination for the graphical information produced by the rendering pipeline,
    a buffer where the rendering output is directed, e.g. framebuffer, texture
    Shader Resource View (SRV): a way to expose a texture to shaders for reading
    A render target view is a way to expose a texture to shaders for writing
    */

    // mpRtOut and pTargetFbo have the same size, i.e. width height of reference frames
    pRenderContext->signal(mpDecodeFence.get(), mNDecodeFenceVal);
    // pRenderContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));
}


int runMain(int argc, char** argv)
{
    unsigned int bitrate = std::stoi(argv[1]);
    unsigned int framerate = std::stoi(argv[2]);
    unsigned int width = std::stoi(argv[3]);
    unsigned int height = std::stoi(argv[4]);
    std::string scene = argv[5];
    unsigned int speedInput = std::stoi(argv[6]);
    std::string scenePath = argv[7];

    // unsigned int width = 1920; // 1920 1280 640
    // unsigned int height = 1080; // 1080 720 360
    // unsigned int bitrate = 5000;
    // unsigned int framerate = 60;
    // std::string scene = "lost_empire";
    // unsigned int speedInput = 1;
    // std::string scenePath = "lost_empire/path1_seg1.fbx"; // no texture, objects are black


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
    encodeDecode.setResolution(width, height);
    encodeDecode.mWidth = width;
    encodeDecode.mHeight = height;
    // encodeDecode.setRefPrefix(scene, speedInput);
    encodeDecode.setDefaultScene(scenePath);
    encodeDecode.setSpeed(speedInput);

    return encodeDecode.run();
}

void foo(EncodeDecode* encodeDecode)
{

    // std::this_thread::sleep_for(std::chrono::duration<float>(100));
    while (true)
    {
        /* code */
        double startTime = 0.0;
        startTime = getCurrentTime(); // Capture the start time
        // std::cout << "Timer started.\n";

        // do the stuff... (decode and display)
        // encodeDecode->encodeFrameBuffer();
        encodeDecode->decodeFrameBuffer();

        double endTime = 0.0;
        endTime = getCurrentTime();

        float elapsedTime = endTime - startTime; // Time since last frame
        std::cout << "elapsedTime: " << elapsedTime << "\n";

        if (elapsedTime < encodeDecode->targetFrameTime)
        {
            // Insert a delay to match the desired FPS
            std::cout << "start sleepoing, targetFrameTime: " << encodeDecode->targetFrameTime << "\n";
            float sleepTime = encodeDecode->targetFrameTime - elapsedTime;
            // sleep(sleepTime); // Sleep for the remaining frame time
            // std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
        }
    }
}


int EncodeDecode::run()
{
    // std::thread decodeThread(foo, this);     // spawn new thread that calls foo()
    return SampleApp::run();
}

int main(int argc, char** argv)
{
    std::cout << "Current Path: " << std::filesystem::current_path() << std::endl;
    // unsigned int bitrate = 3000;
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
