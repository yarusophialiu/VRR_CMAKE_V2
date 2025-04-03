// EncodeDecode.cpp : This file contains the 'main' function. Program execution begins and ends there.
#pragma once

#include <filesystem>
#include <iostream>
#include <d3d12.h>
#include <d3d11.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <wtypesbase.h>
#include <iomanip>
#include <thread>
#include <random>
#include <chrono>
#include <cppcodec/base64_rfc4648.hpp>
#include <span>
#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/TextRenderer.h"
#include "Core/API/NativeHandleTraits.h"
#include "RenderGraph/RenderGraph.h"
#include "slang-gfx.h"

#include "ColorSpace.h"
// #include "NvCodecUtils.h"
#include "EncodeDecode.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvEncodeAPI.h"

#include <Windows.h>
#include <wingdi.h>
// #include "FramePresenterD3D11.h"

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

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

#define THROW_IF_ORT_FAILED(exp) \
    { \
        OrtStatus* status = (exp); \
        if (status != nullptr) \
        { \
            throw ConvertOrtStatusToHResult(*status); \
        } \
    }

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

template<typename T>
using deleting_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

double getCurrentTime()
{
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    auto nowInSeconds = duration_cast<duration<double>>(now.time_since_epoch());
    return nowInSeconds.count();
}


void writeBMP(const char* filename, uint8_t* imageData, int width, int height, int channels=4)
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
    bmpHeader.fileSize = sizeof(BMPHeader) + width * height * channels; // 4 channels (RGBA)
    bmpHeader.dataOffset = sizeof(BMPHeader);
    bmpHeader.headerSize = sizeof(BMPHeader) - 14;
    bmpHeader.width = width;
    bmpHeader.height = -height;
    bmpHeader.planes = 1;
    bmpHeader.bitsPerPixel = 8 * channels;
    bmpHeader.compression = 0;
    bmpHeader.dataSize = width * height * channels;
    bmpHeader.horizontalRes = 2835; // Pixels per meter (72 DPI)
    bmpHeader.verticalRes = 2835;   // Pixels per meter (72 DPI)

    // Write the BMP header
    file.write(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader));
    // Write the pixel data (assuming RGBA format)
    file.write(reinterpret_cast<char*>(imageData), width * height * channels);

    // file.close();
}

void print_vectors(std::vector<float>& fps_vector, std::vector<float>& res_vector)
{
    for (float fps : fps_vector) {
        std::cout << fps << " ";
    }
    std::cout << std::endl;

    for (float res : res_vector) {
        std::cout << res << " ";
    }
    std::cout << std::endl;
}


std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float maxLogit = *std::max_element(logits.begin(), logits.end()); // For numerical stability

    // Compute the exponentials and sum them
    float sumExp = 0.0f;
    for (float logit : logits) {
        sumExp += std::exp(logit - maxLogit); // Subtract maxLogit for stability
    }

    // Compute probabilities
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - maxLogit) / sumExp;
    }

    return probabilities;
}


// Function to write predicted streaming results to a file
void writeToFile(const std::string& filename, int frameNumber, int fps, int resolution) {
    std::ofstream file(filename, std::ios::app);  // Open in append mode
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file << frameNumber << "," << fps << "," << resolution << std::endl;
    file.close();
}


std::string generateTimestampFilename(const std::string& prefix, const std::string& additionalInfo) {
    // Use std::chrono for more modern and efficient time handling
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    // Use std::localtime_s or std::localtime_r for thread-safety
    std::tm localTime = {};
    #ifdef _WIN32
        localtime_s(&localTime, &time);
    #else
        localtime_r(&time, &localTime);
    #endif

    // Use std::format (C++20) or std::stringstream for more efficient formatting
    std::ostringstream filename;
    filename << prefix << "_"
             << additionalInfo << "_"
             << std::put_time(&localTime, "%m%d_%H%M") << ".csv";

    return filename.str();
}


std::pair<int, int> extractBitrateAndSpeed(const std::string& csvFile) {
    // Find the last occurrence of '/' or '\' to isolate the filename
    size_t lastSlash = csvFile.find_last_of("/\\");
    std::string filename = csvFile.substr(lastSlash + 1);

    // Remove the file extension (.csv)
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        filename = filename.substr(0, dotPos);
    }

    // Split the filename by '_'
    std::istringstream ss(filename);
    std::string token;
    std::vector<std::string> parts;
    while (std::getline(ss, token, '_')) {
        parts.push_back(token);
    }

    // Assuming bitrate and speed are in specific positions
    // bitrate is the 4th part (3000kbps), speed is the 3rd part (3)
    if (parts.size() >= 5) {
        int bitrate = std::stoi(parts[3].substr(0, parts[3].find("kbps")));
        int speed = std::stoi(parts[2]);
        return {bitrate, speed};
    }

    // Return -1, -1 if parsing fails
    return {-1, -1};
}


int extractSpeed(const std::string& csvFile) {
    // Find the last occurrence of '/' or '\' to isolate the filename
    size_t lastSlash = csvFile.find_last_of("/\\");
    std::string filename = csvFile.substr(lastSlash + 1);

    // Remove the file extension (.csv)
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        filename = filename.substr(0, dotPos);
    }

    // Split the filename by '_'
    std::istringstream ss(filename);
    std::string token;
    std::vector<std::string> parts;
    while (std::getline(ss, token, '_')) {
        parts.push_back(token);
    }

    // Assuming speed is the 3rd part of the filename
    if (parts.size() >= 3) {
        return std::stoi(parts[2]);
    }

    // Return -1 if parsing fails
    return -1;
}



void saveAsBMP(const std::string& baseFilename, int count, const std::vector<uint8_t>& patchData, int width, int height) {
    std::ostringstream filename;
    filename << baseFilename << "_" << count << ".png";

    // BMP file header (14 bytes)
    uint8_t fileHeader[14] = {
        'B', 'M',                 // Signature
        0, 0, 0, 0,               // File size in bytes (will be filled later)
        0, 0,                     // Reserved
        0, 0,                     // Reserved
        54, 0, 0, 0               // Pixel data offset (54 bytes)
    };

    // BMP info header (40 bytes)
    uint8_t infoHeader[40] = {
        40, 0, 0, 0,              // Header size (40 bytes)
        0, 0, 0, 0,               // Image width (will be filled later)
        0, 0, 0, 0,               // Image height (will be filled later)
        1, 0,                     // Planes (must be 1)
        24, 0,                    // Bits per pixel (24 for RGB)
        0, 0, 0, 0,               // Compression (0 for none)
        0, 0, 0, 0,               // Image size (can be 0 for uncompressed)
        0, 0, 0, 0,               // Horizontal resolution (pixels per meter, ignored)
        0, 0, 0, 0,               // Vertical resolution (pixels per meter, ignored)
        0, 0, 0, 0,               // Colors in color table (0 for none)
        0, 0, 0, 0                // Important color count (0 for all)
    };

    // Calculate padding for each row (rows must be a multiple of 4 bytes)
    int rowSize = (3 * width + 3) & ~3; // Align to 4-byte boundary
    int padding = rowSize - 3 * width;

    // File size and dimensions
    int fileSize = 54 + rowSize * height; // Header + pixel data
    fileHeader[2] = fileSize & 0xFF;
    fileHeader[3] = (fileSize >> 8) & 0xFF;
    fileHeader[4] = (fileSize >> 16) & 0xFF;
    fileHeader[5] = (fileSize >> 24) & 0xFF;

    infoHeader[4] = width & 0xFF;
    infoHeader[5] = (width >> 8) & 0xFF;
    infoHeader[6] = (width >> 16) & 0xFF;
    infoHeader[7] = (width >> 24) & 0xFF;

    infoHeader[8] = height & 0xFF;
    infoHeader[9] = (height >> 8) & 0xFF;
    infoHeader[10] = (height >> 16) & 0xFF;
    infoHeader[11] = (height >> 24) & 0xFF;

    std::ofstream outFile(filename.str(), std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not open file for writing: " << filename.str() << std::endl;
        return;
    }

    // Write headers
    outFile.write(reinterpret_cast<const char*>(fileHeader), 14);
    outFile.write(reinterpret_cast<const char*>(infoHeader), 40);

    // Write pixel data (including padding)
    for (int y = height - 1; y >= 0; --y) { // BMP stores pixels bottom to top
        outFile.write(reinterpret_cast<const char*>(&patchData[y * width * 3]), width * 3);
        outFile.write("\0\0\0", padding); // Write padding bytes
    }

    outFile.close();
    // std::cout << "Image saved to " << filename.str() << std::endl;
}


void readFrameData(const std::string& filename) {
    std::ifstream file(filename);  // Open the file for reading

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {  // Read the file line by line
        std::cout << line << std::endl;  // Print each line to the console
        std::stringstream ss(line);  // Create a stringstream to parse the line
        int nn_count, nn_fps;
        std::string resolution;

        // Parse the line, separating by commas
        char delimiter;  // To handle the commas in the line
        ss >> nn_count >> delimiter >> nn_fps >> delimiter >> resolution;

        // Output the values
        std::cout << "nn_count: " << nn_count << ", nn_fps: " << nn_fps << ", Resolution: " << resolution << std::endl;

    }

    file.close();  // Close the file after reading
}



int argmax(const std::vector<float>& values) {
    return std::distance(values.begin(), std::max_element(values.begin(), values.end()));
}


// Ort::Session get_session(Ort::Env& env, const std::wstring& model_path, Ort::SessionOptions* sessionOptions) {
//     try {
//         // Initialize the session with a wide string path
//         Ort::Session ortSession(env, model_path.c_str(), *sessionOptions);
//         return ortSession;  // Return the session object
//     } catch (const Ort::Exception& e) {
//         // Handle any exceptions related to session creation
//         std::cerr << "Error creating ONNX Runtime session: " << e.what() << std::endl;
//         throw;  // Re-throw the exception if needed
//     }
// }

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

// static const Falcor::float4 kClearColor(0.38f, 0.52f, 0.10f, 1);
// static const Falcor::float4 kClearColor(0.5f, 0.16f, 0.098f, 1);
static const Falcor::float4 kClearColor(0.f, 0.f, 0.0f, 1);


// constructor
EncodeDecode::EncodeDecode(const SampleAppConfig& config) : SampleApp(config)
{
        mConfig = config;
        mWidth = config.windowDesc.width;   // 1920, 4096, 1280, 854, 640, 960, 1024, 1280, 1440, 1423
        mHeight = config.windowDesc.height; // 1080, 2160, 720, 480, 360, 540, 600, 800, 900, 800


        stimuliState = vrrON == true ? stimuli_state_t::DROPJOD04 : stimuli_state_t::DROPJOD06;
        mpDevice = getDevice();
        mpD3D12Device = mpDevice->getNativeHandle().as<ID3D12Device*>();

        mpDecodeFence = mpDevice->createFence();
        mNInputFenceVal = 0;
        mNOutputFenceVal = 0;
        mNDecodeFenceVal = 0;

        mNVEnc = {};
        mEBufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
        mCudaDevice = 0;
        // mpRtOut = getDevice()->createTexture2D(mWidth1920, mHeight1080, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        mpRtOut = getDevice()->createTexture2D(mWidth, mHeight, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        // std::cout << "bitrate: " << bitRate << std::endl;

        // cast into directx 12 using: ->getNativeHandle().as<ID3D12Resource*>();
        // falcor's device, createtexture3d
        //mPDecoderOutputTexture1080 = getDevice()->createTexture2D(1920, 1080, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture360 = getDevice()->createTexture2D(640, 360, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture480 = getDevice()->createTexture2D(854, 480, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture720 = getDevice()->createTexture2D(1280, 720, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture864 = getDevice()->createTexture2D(1536, 864, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        mPEncoderInputTexture1080 = getDevice()->createTexture2D(1920, 1080, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);

        //mpD3D12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mpCommandAllocator));

        // motion vector mipmap
        mipLevels = 1; // 7: 64x64 patches, 11: 1x1 patch, 1: 1920x1080 patch
        std::cout << "constructor mipLevels: " << mipLevels << "\n";
        mpRtMip = getDevice()->createTexture2D(
            mWidth1920, mHeight1080, ResourceFormat::RG32Float, 1, mipLevels, nullptr,
            // mWidth, mHeight, ResourceFormat::RG32Float, 1, mipLevels, nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget
        );

        initPairList();
        if (!mConditions.empty()) {
            mCurrentCondition = mConditions[0];
        } else {
            std::cerr << "Error: mConditions is empty! No conditions were generated.\n";
        }
        // mCurrentCondition = mConditions[0];
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

    // set bitrate, scene for the first pair
    std::cout << "onloading mCurrentTrial " << mCurrentTrial << std::endl;
    mCurrentCondition = mConditions[mCurrentTrial];

    initEncoder();
    initDecoder();
    std::cout << "load scene: " << std::endl;

    // readFrameData("C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/nn_results.txt");
    loadScene(kDefaultScene, getTargetFbo().get());

    Properties gBufferProps = {};
    Properties fXAAProps = {};
    Properties computeVelocityProps = {};

    // RenderGraph有DirectedGraph, DirectedGraph 存储了 PassId 和 EdgeId 的关系，
    // 而 RenderGraph 则存储了两个 Id 所指向的资源。而这些关系的产生都在 addPass 和 addEdge 中完成
    mpRenderGraph = RenderGraph::create(mpDevice, "EncodeDecode");
    mpRenderGraph->createPass("GBuffer", "GBufferRaster", gBufferProps);
    mpRenderGraph->createPass("TAA", "TAA", fXAAProps);
    // mpRenderGraph->createPass("ComputeVelocity", "ComputeVelocity", computeVelocityProps);

    mpRenderGraph->onResize(getTargetFbo().get());
    mpRenderGraph->setScene(mpScene);
    mpRenderGraph->addEdge("GBuffer.mvec", "TAA.motionVecs"); // source texture, output texture
    // mpRenderGraph->addEdge("GBuffer.mvec", "ComputeVelocity.motionVecs"); // source texture, output texture
    mpRenderGraph->markOutput("TAA.colorOut");
    mpRenderGraph->markOutput("GBuffer.mvec"); // Additional motion vector info
    mpRenderGraph->setInput("TAA.colorIn", mpRtOut);
    mpComputeVelocityPass = ComputePass::create(mpDevice, "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/ComputeVelocity.cs.slang", "computeVelocity", DefineList());

    mpRenderGraph->compile(pRenderContext, log);

    // Make output buffer to store ptch motion average
    mpVelocity = mpDevice->createBuffer(4 * sizeof(float), ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal);
    mpPatchTexture = mpDevice->createTexture2D(patchWidth, patchHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);

    // allocate memory so encoder can work with what we need
    makeEncoderInputBuffers(6);
    makeEncoderOutputBuffers(1);
}

void EncodeDecode::onResize(uint32_t width, uint32_t height)
{
    float h = (float)height; // 1080, 2160, 720,
    float w = (float)width;  // 1920, 3840, 1280,

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
void EncodeDecode::extract_patch_from_frame(std::vector<uint8_t>& renderedFrameVal, uint32_t frameWidth, uint32_t frameHeight, \
                                uint32_t startX, uint32_t startY, std::vector<uint8_t>& patchData)
{
    uint32_t numChannels = 4;  // For BGRA8 format (8 bits per channel, 4 channels)
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


// Function to decide if settings should change, output fps, resolution based on probability
std::tuple<int, int, int> EncodeDecode::shouldChangeSettings(int currentFps, int currentResolution, std::vector<float>& fps_probabilities,  std::vector<float>& res_probabilities) {
    // Adjust probabilities based on currentResolution
    std::vector<float> p_res(5);
    std::vector<float> p_fps(10);
    float resolutionBias = 0.0f;
    float framerateBias = 0.0f;

    for (int index = 0; index < res_probabilities.size(); ++index) {
        int resolution = reverse_res_map[index];
        if (resolution == currentResolution) {
            p_res[index] = resolutionBias + res_probabilities[index];
        } else {
            p_res[index] = res_probabilities[index];
        }
    }

    for (int index = 0; index < fps_probabilities.size(); ++index) {
        int fps = reverse_fps_map[index];
        if (fps == currentFps) {
            p_fps[index] = framerateBias + fps_probabilities[index];
        } else {
            p_fps[index] = fps_probabilities[index];
        }
    }

    // Find the maximum resolution probability
    auto max_res_it = std::max_element(p_res.begin(), p_res.end());
    int max_res_index = std::distance(p_res.begin(), max_res_it);
    float max_res_value = *max_res_it; // the value is the probability

    // Find the maximum FPS probability
    auto max_fps_it = std::max_element(p_fps.begin(), p_fps.end());
    int max_fps_index = std::distance(p_fps.begin(), max_fps_it);
    float max_fps_value = *max_fps_it;

    // if the max value is greater than 1, choose it as selected fps/res
    // if the selected is different from the current one, reset the fps/res
    std::cout << "p_fps, p_res now " << std::endl;
    print_vectors(p_fps, p_res);

    std::cout << "Max FPS index: " << max_fps_index << ", value: " << max_fps_value << std::endl;
    std::cout << "Max resolution index: " << max_res_index << ", value: " << max_res_value << std::endl;

    int h = reverse_res_map[max_res_index];
    int w = res_map_by_height[h];

    return std::make_tuple(reverse_fps_map[max_fps_index], w, h);
}



// bool EncodeDecode::getProbabilitiesForFrame(int frameNumber, std::vector<float>& resProbabilities, std::vector<float>& fpsProbabilities) {
//     // Find the index of the given frameNumber
//     auto it = std::find(frameNumbersCSV.begin(), frameNumbersCSV.end(), frameNumber);
//     if (it != frameNumbersCSV.end()) {
//         size_t index = std::distance(frameNumbersCSV.begin(), it);

//         // Retrieve probabilities for the frame
//         resProbabilities = resProbabilitiesCSV[index];
//         fpsProbabilities = fpsProbabilitiesCSV[index];
//         return true; // Success
//     } else {
//         std::cerr << "Frame number " << frameNumber << " not found." << std::endl;
//         return false; // Frame number not found
//     }
// }


void EncodeDecode::readCsv(const std::string& filename, int& resolution, int& fps)
{

    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        // exit(0);
        return;
    }

    std::string line;
    if (std::getline(file, line)) {  // Read only the first line
        std::istringstream iss(line);
        std::string fpsStr, resolutionStr;

       if (std::getline(iss, fpsStr, ',') && std::getline(iss, resolutionStr, ',')) {
            try {
                fps = std::stoi(fpsStr);
                resolution = std::stoi(resolutionStr);
            } catch (const std::exception& e) {
                std::cerr << "Error converting to integer: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Error: Could not extract fps and resolution." << std::endl;
        }

        std::cout << "Read first line values: fps = " << fps << ", resolution = " << resolution << std::endl;
    }

    file.close();
}



// sequences picked by Plot_HPC pick_sequences.py
void EncodeDecode::initPairList() {
    // max jod vs 0.25
    manager.addStimulus(ExperimentStimulus("living_room", "path2_seg3_1", 2000, 720, 90, 1.f, 0.4f, 0));
    manager.addStimulus(ExperimentStimulus("living_room", "path2_seg3_1", 2000, 1080, 120, 1.f, 0.6f, 0));

    manager.generateConditions(mConditions);
    std::cout << "mConditions size " << mConditions.size() << "\n";

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);
    std::shuffle(mConditions.begin(), mConditions.end(), eng);
}


std::string EncodeDecode::getInferencePathNameForStimulus(ExperimentStimulus* stimulus) {

    if (stimulus->speed == 1) {
        return stimulus->DROPJODSCALE == 0.4f
            ? "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/nnOutput/drop-jod/patch-data-dropjod04-1-CSV/"
            : "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/nnOutput/drop-jod/patch-data-dropjod06-1-CSV/";
    }
    else if (stimulus->speed == 2) {
        return stimulus->DROPJODSCALE == 0.4f
            ? "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/nnOutput/drop-jod/patch-data-dropjod04-2-CSV/"
            : "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/nnOutput/drop-jod/patch-data-dropjod06-2-CSV/";
    } else if (stimulus->speed == 3) {
        return stimulus->DROPJODSCALE == 0.4f
            ? "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/nnOutput/drop-jod/patch-data-dropjod04-3-CSV/"
            : "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/nnOutput/drop-jod/patch-data-dropjod06-3-CSV/";
    }

    return "";
}


void EncodeDecode::switchToNextPair() {

    ++mCurrentTrial;

    std::cout << "Switching pair to condition: " << mCurrentTrial << " of " << mConditions.size() << "\n";

    mCurrentCondition = mConditions[mCurrentTrial];


    // std::string inferenceDir1 = getInferencePathNameForStimulus(&mCurrentCondition.stimulus1);
    // std::string inferenceFileName1 = getInferenceFileNameForStimulus(&mCurrentCondition.stimulus1);
    // std::string inferenceFilePath1 = inferenceDir1 + inferenceFileName1;
    // readCsv(inferenceFilePath1, mCurrentCondition.stimulus1.resolution, mCurrentCondition.stimulus1.framerate);

    // std::string inferenceDir2 = getInferencePathNameForStimulus(&mCurrentCondition.stimulus2);
    // std::string inferenceFileName2 = getInferenceFileNameForStimulus(&mCurrentCondition.stimulus2);
    // std::string inferenceFilePath2 = inferenceDir2 + inferenceFileName2;
    // readCsv(inferenceFilePath2, mCurrentCondition.stimulus2.resolution, mCurrentCondition.stimulus2.framerate);

    std::cout << "mCurrentTrial " << mCurrentTrial << std::endl;

    // TODO Why set it here?
    selectedSpeed = speed;
    selectedFps = frameRate;
    selectedHeight = mHeight;
    selectedWidth = res_map_by_height[mHeight];
    std::cout << "selectedFps: " << selectedFps << " selectedHeight " << selectedHeight << " selectedWidth " << selectedWidth << "\n";


    setScene(mCurrentCondition.stimulus1.sceneIndex);
    setSpeed(mCurrentCondition.stimulus1.speed);
    setResolution(1920, 1080);
    setFrameRate(60);
    setBitRate(mCurrentCondition.stimulus1.bitrate);
    mTimeSecs = 0;
    mTimeFrames = 0;

    mOldWidth = 1920;
    mOldHeight = 1080;

    std::srand(std::time(0));
    bothViewed = false;
    switchAB = (std::rand() % 2) == 0;

    if (!switchAB) {
        vrrON = true;
        stimuliState = stimuli_state_t::DROPJOD04;
    } else {
        vrrON = false;
        stimuliState = stimuli_state_t::DROPJOD06;
        resetBaseline = true;
    }
    // std::this_thread::sleep_for(std::chrono::seconds(8));
}


void EncodeDecode::appendChoiceToCsv()
{
    fs::path dirPath = fs::path(szExperimentPrefixFilePath) / experimentFilename;
    // Open the file in append mode
    std::ofstream file(dirPath, std::ios::app);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file for appending." << std::endl;
        return;
    }


    file << observerId << "," << mCurrentCondition.stimulus1.sceneName << "," << mCurrentCondition.stimulus1.bitrate << "," << speed  << "," << ((vrrON)? "1" : "0") << "," << (stimuliState == stimuli_state_t::DROPJOD04? "0.4" : "0.6") << "," << mHeight << "," << frameRate << "\n"; // End of the row
    file.close();
}


void EncodeDecode::appendRowToCsv(int frameNumber,
                    const std::vector<float>& res_probabilities, const std::vector<float>& fps_probabilities)
{
    std::string csvName = mCurrentCondition.stimulus1.sceneName + "_" + mCurrentCondition.stimulus1.pathName
                        + "_" + std::to_string(mCurrentCondition.stimulus1.bitrate) + "kbps.csv";
    fs::path dirPath = fs::path(szNNOutputPrefixFilePath) / csvName;
    // Open the file in append mode
    std::ofstream file(dirPath, std::ios::app);
    // std::ofstream file(filename, std::ios::app); // Open in append mode

    if (!file.is_open()) {
        std::cerr << "Failed to open the file for appending." << std::endl;
        return;
    }

    file << frameNumber << ",";

    // Write resolution probabilities
    for (size_t i = 0; i < res_probabilities.size(); ++i) {
        file << res_probabilities[i];
        if (i != res_probabilities.size() - 1) {
            file << " "; // Separate probabilities with spaces
        }
    }
    file << ",";

    // Write FPS probabilities
    for (size_t i = 0; i < fps_probabilities.size(); ++i) {
        file << fps_probabilities[i];
        if (i != fps_probabilities.size() - 1) {
            file << " "; // Separate probabilities with spaces
        }
    }
    file << ",";
    file << frameRate;
    file << ",";
    file << mHeight;


    file << "\n"; // End of the row
    file.close();
}


void EncodeDecode::testKeyChange()
{
    if (mResolutionChange == -1)
    {
        setResolution(854, 480);
        mResolutionChange = 0;
    } else if (mResolutionChange == 1) {

        setResolution(1920, 1080);
        mResolutionChange = 0;
    } else if (mResolutionChange == -2) {

        setResolution(640, 360);
        mResolutionChange = 0;
    }
}


float EncodeDecode::computePatchVelocity(RenderContext* pRenderContext, int startX, int startY)
{
    auto rootVar = mpComputeVelocityPass->getRootVar();
    rootVar["gInputImage"] = mpRenderGraph->getOutput("GBuffer.mvec")->asTexture();
    rootVar["gOutputVelocity"] = mpVelocity;
    auto globalVar = rootVar["PerFrameCB"];
    globalVar["gPatchOffset"] = Falcor::uint2(startX, startY); // top left coordinate of patch
    mpComputeVelocityPass->execute(pRenderContext, Falcor::uint3(1u, 1u, 1u));

    pRenderContext->submit();
    float patchVelocity = mpVelocity->getElements<float>(0, 1).at(0); // compute motion velocity in pixel per degree in terms of fps166
    // std::cout << "mpVelocity: " << patchVelocity << "\n"; // starting index, the number of elements
    // std::cout << "frameRate, " << frameRate << ", resolution " << mHeight << "\n"; // starting index, the number of elements
    return patchVelocity;
}


// void EncodeDecode::runONNXInference(RenderContext* pRenderContext, int startX, int startY, float patchVelocity,
//                                     std::vector<float>& outputResTensorValues, std::vector<float>& outputFpsTensorValues)
// {
//     ref<Texture> frameTexture = mpRenderGraph->getOutput("TAA.colorOut")->asTexture(); // GBuffer.mvec
//     std::vector<uint8_t> renderedFrameVal = pRenderContext->readTextureSubresource(frameTexture.get(), 0);
//     std::vector<uint8_t> patchData(patchWidth * patchHeight * 3);
//     uint32_t frameWidth = frameTexture->getWidth();
//     uint32_t frameHeight = frameTexture->getHeight();
//     // renderedFrameVal has 4 channels, patch has 3 channels

//     extract_patch_from_frame(renderedFrameVal, 1920, 1080, startX, startY, patchData);

//     std::string dirName = mCurrentCondition.stimulus1.sceneName + "_" + std::to_string(mCurrentCondition.stimulus1.pathIndex) + "_" + std::to_string(mCurrentCondition.stimulus1.bitrate) + "_" + std::to_string(speed);
//     std::filesystem::create_directory(dirName);

//     std::string fileName = dirName + "/" + std::to_string(mTimeFrames) + ".bmp";
//     writeBMP(fileName.c_str(), patchData.data(), patchWidth, patchHeight, 3);
//     FILE* fp = nullptr;


//     if (mTimeFrames == 2) {

//         fp = fopen((dirName + "/" + "velocities.csv").c_str(), "w+");
//     } else {

//         fp = fopen((dirName + "/" + "velocities.csv").c_str(), "a+");
//     }

//     std::string velocity_str = std::to_string(patchVelocity) + "\n";
//     fprintf(fp, velocity_str.c_str());

//     fclose(fp);


//     // fileName = "frame_" + std::to_string(mTimeFrames) + ".png";

//     // frameTexture->captureToFile(0, 0, fileName);

//     std::vector<float> floatData(patchData.size()); // how many channels? 49152 = 128x128x3

//     // Normalize each uint8_t to a float in the range [0, 1]
//     float patchMean = 0;
//     for (size_t i = 0; i < patchData.size(); ++i) {
//         floatData[i] = static_cast<float>(patchData[i]) / 255.0f;
//         patchMean += floatData[i];
//     }

//     patchMean /= patchData.size();

//     std::cout << "Patch mean: " << patchMean << "\n";

//     int64_t fps = 166;
//     std::vector<int64_t> fpsVec = {fps};
//     std::vector<int64_t> bitrateVec = {targetBitrate}; // TODO: change for bitrate you want
//     std::vector<int64_t> resolutionVec = {1080};
//     std::vector<float> velocityVec = {patchVelocity};

//     std::cout << "Target bitrate: " << targetBitrate << "\n";
//     std::cout << "Motion velocity: " << patchVelocity << "\n";

//     Ort::AllocatorWithDefaultOptions allocator;
//     Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

//     std::vector<int64_t> inputDims = {1, 3, patchHeight, patchWidth};
//     std::vector<int64_t> scalerInputDims = {1};
//     Ort::Value inputImageTensor = Ort::Value::CreateTensor<float>(memoryInfo, floatData.data(), floatData.size(), inputDims.data(), inputDims.size());
//     Ort::Value inputFpsTensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, fpsVec.data(), fpsVec.size(), scalerInputDims.data(), scalerInputDims.size());
//     Ort::Value inputBitrateTensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, bitrateVec.data(), bitrateVec.size(), scalerInputDims.data(), scalerInputDims.size());
//     Ort::Value inputResolutionTensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, resolutionVec.data(), resolutionVec.size(), scalerInputDims.data(), scalerInputDims.size());
//     Ort::Value inputVelocityTensor = Ort::Value::CreateTensor<float>(memoryInfo, velocityVec.data(), velocityVec.size(), scalerInputDims.data(), scalerInputDims.size());

//     std::vector<int64_t> outputDims = {1, 5};
//     std::vector<int64_t> fpsOutputDims = {1, 10};
//     size_t outputResTensorSize = 5; // vectorProduct(outputDims);
//     size_t outputFpsTensorSize = 10; // vectorProduct(outputDims);

//     Ort::Value outputResTensor = Ort::Value::CreateTensor<float>(memoryInfo, outputResTensorValues.data(), outputResTensorSize, outputDims.data(), outputDims.size());
//     Ort::Value outputFpsTensor = Ort::Value::CreateTensor<float>(memoryInfo, outputFpsTensorValues.data(), outputFpsTensorSize, fpsOutputDims.data(), fpsOutputDims.size());

//     auto bindings = Ort::IoBinding::IoBinding(*ortSession);
//     bindings.BindInput(ortSession->GetInputNameAllocated(0, ortAllocator).get(), inputImageTensor);
//     bindings.BindInput(ortSession->GetInputNameAllocated(1, ortAllocator).get(), inputFpsTensor);
//     bindings.BindInput(ortSession->GetInputNameAllocated(2, ortAllocator).get(), inputBitrateTensor);
//     bindings.BindInput(ortSession->GetInputNameAllocated(3, ortAllocator).get(), inputResolutionTensor);
//     bindings.BindInput(ortSession->GetInputNameAllocated(4, ortAllocator).get(), inputVelocityTensor);
//     bindings.BindOutput(ortSession->GetOutputNameAllocated(0, ortAllocator).get(), outputResTensor);
//     bindings.BindOutput(ortSession->GetOutputNameAllocated(1, ortAllocator).get(), outputFpsTensor);

//     std::cout << "Running ONNX model\n";
//     Ort::RunOptions runOpts;
//     runOpts.SetRunLogVerbosityLevel(3); // 0 = Default, higher values mean more detailed logging
//     ortSession->Run(runOpts, bindings);
//     bindings.SynchronizeOutputs();

//     print_vectors(outputFpsTensorValues, outputResTensorValues);
// }


int findMaxIndex(const std::vector<float>& probabilities) {
    auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
    return std::distance(probabilities.begin(), maxIt);
}


// normalize outputtensors, then apply softmax
void EncodeDecode::processNNOutput(std::vector<float>& outputResTensorValues,
                                    std::vector<float>& outputFpsTensorValues,
                                    std::vector<float>& res_probabilities,
                                    std::vector<float>& fps_probabilities)
{
    // Find predictions
    int res_pred_index = argmax(outputResTensorValues);
    int fps_pred_index = argmax(outputFpsTensorValues);
    int predicted_resolution = reverse_res_map[res_pred_index];
    int predicted_fps = reverse_fps_map[fps_pred_index];
    std::cout << "fps_preds predicted_fps: " << predicted_fps << " fps" << std::endl;
    std::cout << "res_preds predicted_resolution: " << predicted_resolution << "p" << std::endl;

    float maxRes = *std::max_element(outputResTensorValues.begin(), outputResTensorValues.end());
    for (float& value : outputResTensorValues) {
        value /= maxRes;
    }

    float maxFps = *std::max_element(outputFpsTensorValues.begin(), outputFpsTensorValues.end());
    for (float& value : outputFpsTensorValues) {
        value /= maxFps;
    }

    print_vectors(outputResTensorValues, outputFpsTensorValues);


    // // std::cout << "current fps: " << frameRate << ", current resolution: " << mHeight << " fps" << std::endl;
    // saveAsBMP("patchData", fCount_rt, patchData, patchWidth, patchHeight); // C:\Users\15142\new\Falcor\build\windows-vs2022\bin\Debug

    res_probabilities = softmax(outputResTensorValues); // outputResTensorValues is processed
    fps_probabilities = softmax(outputFpsTensorValues);
    // std::cout << "res_probabilities, fps_probabilities" << " ";
}


void EncodeDecode::generateSettings()
{
    std::vector<int> bitrates = {1000, 1500, 2000, 3000, 4000};
    std::vector<int> framerates = {30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    std::vector<std::pair<int, int>> resolutions = {
        {1920, 1080},
        {1536, 864},
        {1280, 720},
        {854, 480},
        {640, 360}
    };

    for (int b : bitrates)
    {
        for (auto [w, h] : resolutions)
        {
            for (int f : framerates)
            {
                mSettings.emplace_back(f, w, h, b);
            }
        }
    }
}



// called in sampleapp renderframe()
void EncodeDecode::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    double startTime = 0.0;
    startTime = getCurrentTime(); // Capture the start time
    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    static double timeSecs = 0; // timeSecs is the time through animation, i.e. camera path
    if (mpScene)
    {
        if (timeSecs > 2)
        {
            // TODO: reset framerate, bitrate, resolution, speed

            fpEncOut->close(); // end h265 file but the rendering continues
        }
        Scene::UpdateFlags updates = mpScene->update(pRenderContext, mCurrentCondition.stimulus1.speed * timeSecs); // 2* timesec, 0.5
        // Scene::UpdateFlags updates = mpScene->update(pRenderContext, mCurrentCondition.stimulus1.speed * ((double)mTimeFrames / frameRate)); // 2* timesec, 0.5
        std::cout << "Scene animation duration(s): " << mpScene->getAnimationDurationSecs() << "\n";
        if (is_set(updates, Scene::UpdateFlags::GeometryChanged))
            FALCOR_THROW("This sample does not support scene geometry changes.");
        if (is_set(updates, Scene::UpdateFlags::RecompileNeeded))
            FALCOR_THROW("This sample does not support scene changes that require shader recompilation.");

        static uint32_t fcount = 0;
        static int fCount_rt = 0;
        std::cout << "fCount_rt " << fCount_rt << "\n";
        std::cout << "timeSecs " << timeSecs << "\n";
        std::cout << "mTimeFrames " << mTimeFrames << ", mTimeFrames / frameRate" << (double)mTimeFrames / frameRate << "\n";
        std::cout << "fcount " << fcount << "\n";
        // std::cout << "frameLimit " << frameLimit << "\n";

        mpRenderGraph->execute(mpRenderContextDecode);
        InterlockedIncrement(&mNDecodeFenceVal);

        if (mRayTrace)
            renderRT(mpRenderContextDecode, pTargetFbo, mWidth, mHeight);
        else
            renderRaster(pRenderContext, pTargetFbo);

        // // blit from one frame buffer (or texture) to another
        // // displaying the final rendered image to the screen, framebuffer represents the final render target, usually the screen or a backbuffer
        // // pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), pTargetFbo->getRenderTargetView(0));

        cpuWaitForFencePoint(mpDecodeFence->getNativeHandle().as<ID3D12Fence*>(), mNDecodeFenceVal);
        // blit TAA.colorOut into smaller texture, want to encode smaller texture

        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture360->getRTV(0));
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture480->getRTV(0));
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture720->getRTV(0));
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture864->getRTV(0));
        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), mPEncoderInputTexture1080->getRTV(0));

        if (outputReferenceFrames && (fCount_rt > 0))
        {
            snprintf(szRefOutFilePath, sizeof(szRefOutFilePath), "%s%d.bmp", refBaseFilePath, fCount_rt);
            // mpRtOut->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);
            mpRenderGraph->getOutput("TAA.colorOut")->asTexture()->captureToFile(0, 0, szRefOutFilePath, Bitmap::FileFormat::BmpFile, Bitmap::ExportFlags::None, false);

            // std::string path = refBaseFilePath + std::to_string(mTimeFrames) + ".png";
            // mpRenderGraph->getOutput("TAA.colorOut")->asTexture()->captureToFile(0, 0, path, Bitmap::FileFormat::PngFile);
        }

        pRenderContext->blit(mpRenderGraph->getOutput("TAA.colorOut")->getSRV(), pTargetFbo->getRenderTargetView(0));

        // if (mTimeFrames >= 2)
        if (fCount_rt >= 1)  // 2
        {
            // std::cout << "\nframeCount: " << mTimeFrames << "\n";
            std::cout << "bitrate : " << bitRate << "\n";
            std::cout << "resolution : " << mWidth << "x" << mHeight << "\n";
            std::cout << "framerate : " << frameRate << "\n";
            std::cout << "speed : " << speed << "\n\n";

            encodeFrameBuffer();

            if (frameLimit > 0 && fcount >= frameLimit)
            // if (mTimeFrames > 30)
            {
                std::exit(0);
            }
            timeSecs += 1.0 / frameRate; // disable line 520 about update timeSecs
        }

        // if (fCount_rt == 20)
        // {
        //     setResolution(854, 480);
        //     // setFrameRate(10);
        // }

        fCount_rt += 1;
        ++fcount;
        mTimeFrames += 1;
        mTimeSecs += 1.0 / frameRate;
    }


    // stimuliString += (vrrON ? " vrr ON\n" : " vrr OFF\n");
    // getTextRenderer().render(pRenderContext, stimuliString, pTargetFbo, {20, 20});

    mOldWidth = mWidth;
    mOldHeight = mHeight;
    ++currentResolutionFrameLength;
    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});

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

    if (keyEvent.type != KeyboardEvent::Type::KeyPressed)
        return false;

    if (keyEvent.key == Falcor::Input::Key::Left) { // stimulus 1, drop-jod 0.4
        // mResolutionChange = 1;
        if (!switchAB) {
            vrrON = true;
            stimuliState = stimuli_state_t::DROPJOD04;
        } else { // stimulus 2, drop-jod 0.6
            vrrON = false;
            stimuliState = stimuli_state_t::DROPJOD06;
            resetBaseline = true;
        }

        mTimeFrames = 0;
        mTimeSecs = 0;
        switchCondition = true;
        return true;
    } else if (keyEvent.key == Falcor::Input::Key::Right) {
        // change resolution and framerate to 1080p 60?
        // mResolutionChange = -1;

        if (switchAB) {
            vrrON = true;
            stimuliState = stimuli_state_t::DROPJOD04;
        } else {
            vrrON = false;
            stimuliState = stimuli_state_t::DROPJOD06;
            resetBaseline = true;
        }

        mTimeFrames = 0;
        mTimeSecs = 0;
        switchCondition = true;
        bothViewed = true;
        //currentSceneIdx += 1;
        return true;
    } else if (keyEvent.key == Falcor::Input::Key::Space && bothViewed) {
        //Record choice here experimentFilename
        std::cout << "mCurrentTrial " << mCurrentTrial << std::endl;
        appendChoiceToCsv();
        pairIndex += 1; // jump to next pair

        if (mCurrentTrial >= mConditions.size() - 1) {

            std::cout << "Exiting on space\n";
            exit(0);
            return false;
        }

        mSwitchOnNext = 1;
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
    pIntializeParams->enableEncodeAsync = 1; // TODO: async 0: async = 0 is sync mode, 1 is async mode

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
    // mEncodeConfig.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
    mEncodeConfig.encodeCodecConfig.hevcConfig.idrPeriod = NVENC_INFINITE_GOPLENGTH;

    mEncodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR; // NV_ENC_PARAMS_RC_VBR NV_ENC_PARAMS_RC_CONSTQP, NV_ENC_PARAMS_RC_CBR,
    mEncodeConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION; // not valid for h264
    // disable below for CRF (target quality)
    mEncodeConfig.rcParams.averageBitRate = bitRate * 1000;
    // mEncodeConfig.rcParams.averageBitRate = 0;
    mEncodeConfig.rcParams.maxBitRate = bitRate * 1000;
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
            std::string sceneFolder = sceneFull.substr(0, sceneFull.find('/')); // e.g. lost_empire
            std::cout << "\n\n\nScene name: " << sceneName << std::endl;
            newFilePath << "encodedH264/"
                << dateStr << "/"
                << sceneFolder << "/"
                << sceneNameOnly << "_" << speed << "/"
                << bitRate << "/"
                << bitRate << "_" << frameRate << "_" << mHeight << ".h265";

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
    picParams.inputWidth = mOldWidth; // TODO: should it be 1080?
    picParams.inputHeight = mOldHeight;
    picParams.outputBitstream = mVOutputRsrc[0];
    picParams.completionEvent = mVPCompletionEvent[bufferIndex];

    // switch condition when changing between our technique and standard technique
    if (switchCondition || mTimeFrames == 0) {
        picParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR; // Force an I-frame so we don't use content from the other method to generate our frames.
        switchCondition = false;
    }
    // encoded results written to mVOutputRsrc
    NVENCSTATUS nvStatus = mNVEnc.nvEncEncodePicture(mHEncoder, &picParams);

    // std::cout << "Endcoding dimensions: " << mWidth << "x" << mHeight << "\n";

    waitForCompletionEvent(bufferIndex); // wait for nvEncEncodePicture to finish

    // write encoded frames to out_.h264
    if (outputEncodedFrames)
    {
        // std::cout << "write encoded h264 to: " << szOutFilePath << std::endl;
        fpEncOut->write(reinterpret_cast<char*>(mVEncodeOutData.data()), mVEncodeOutData.size());
    }

    mVEncodeOutData.clear(); // clear the previous encoded frame

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
    pInitializeParams->ulMaxWidth = 1920;// mWidth;
    pInitializeParams->ulMaxHeight = 1080;// mHeight;
    pInitializeParams->ulTargetWidth = mWidth;
    pInitializeParams->ulTargetHeight = mHeight;
    pInitializeParams->enableHistogram = 0;
}


void EncodeDecode::initDecoder()
{

    std::cout << "Initialising decoder\n";

    // Initialise CUDA first, check how many gpus we have, use the first one
    cuInit(0);

    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);

    std::cout << "Using CUDA version: " << cudaVersion << "\n";

    std::cout << "Getting GPU devices\n";

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
    char szDeviceName[256];
    cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
    //std::cout << "GPU in use: " << szDeviceName << "\n";
    std::cout << "Making CUDA context\n";
    cuDeviceGet(&mCudaDevice, 0);
    std::cout << "Making CUDA context\n";
    CUresult contextError = cuCtxCreate(&mCudaContext, CU_CTX_SCHED_BLOCKING_SYNC, mCudaDevice);

    if (contextError != CUDA_SUCCESS) {

        std::cout << "Error making CUDA context\n";
    }

    NVDEC_API_CALL(cuvidCtxLockCreate(&mCtxLock, mCudaContext));

    std::cout << "Making CUDA stream\n";
    cuStreamCreate(&mCuvidStream, CU_STREAM_DEFAULT);

    /*
    a parser is a component responsible for analyzing the bitstream or input data and extracting relevant information or structures
    parse the bitstream and perform decoding
    */
    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = cudaVideoCodec_HEVC; // cudaVideoCodec_HEVC, cudaVideoCodec_H264
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

    std::cout << "Creating video parser";

    NVDEC_API_CALL(cuvidCreateVideoParser(&mHParser, &videoParserParameters));

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

    std::cout << "Creating decoder";
    // Make the decoder
    CUVIDDECODECREATEINFO videoDecodeCreateInfo = {0};
    makeDefaultDecodingParams(&videoDecodeCreateInfo);

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext));
    NVDEC_API_CALL(cuvidCreateDecoder(&mHDecoder, &videoDecodeCreateInfo)); // create decoder
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(nullptr));
    makeDecoderOutputBuffers(); // allocate cuda memory
}


#include "WindowsSecurityAttributes.h"

void EncodeDecode::makeDecoderOutputBuffers()
{
    std::cout << "Making decoder output buffers\n";

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = 1920; // mWidth; 1920
    resourceDesc.Height = 1080; // mHeight; 1080
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    //resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    resourceDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;

    std::cout << "Making decoder output texture\n";

    //mPDecoderOutputTexture1080 = getDevice()->createTexture2D(1920, 1080, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::Shared);
    //ID3D12Resource* dx12Texture = mPDecoderOutputTexture1080->getNativeHandle().as<ID3D12Resource*>();
    ID3D12Resource* dx12Texture;
    mpD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE | D3D12_HEAP_FLAG_SHARED, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&dx12Texture));
    D3D12_RESOURCE_DESC dx12ResourceDesc = dx12Texture->GetDesc();

    gfx::InteropHandle interopHandle = {gfx::InteropHandleAPI::D3D12, (uint64_t)dx12Texture};
    gfx::ITextureResource* gfxTextureResource;
    gfx::ITextureResource::Desc gfxTextureDesc = {};

    gfxTextureDesc.defaultState = gfx::ResourceState::General;
    gfxTextureDesc.memoryType = gfx::MemoryType::DeviceLocal;
    gfxTextureDesc.size.width = 1920;
    gfxTextureDesc.size.height = 1080;
    gfxTextureDesc.size.depth = 1;
    gfxTextureDesc.arraySize = 1;
    gfxTextureDesc.numMipLevels = 1;
    gfxTextureDesc.format = gfx::Format::B8G8R8A8_UNORM; // lookup can result in Unknown / unsupported format
    gfxTextureDesc.sampleDesc.numSamples = 1;
    gfxTextureDesc.sampleDesc.quality = 0;

    mpDevice->getGfxDevice()->createTextureFromNativeHandle(interopHandle, gfxTextureDesc, &gfxTextureResource);

    mPDecoderOutputTexture1080 = make_ref<Texture>(
        Texture(mpDevice, gfxTextureResource,
        Resource::Type::Texture2D, ResourceFormat::BGRA8Unorm,
        1920, 1080, 1, 1, 1, 1,
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::Shared,
        Resource::State::Common)
    );

    std::cout << "Decorder texture layout is: " << dx12ResourceDesc.Layout << "\n";

    std::cout << "Making shared handle\n";

    HANDLE sharedHandle;
    WindowsSecurityAttributes windowsSecurityAttributes;
    LPCWSTR name = NULL;
    HRESULT result;
    result = mpD3D12Device->CreateSharedHandle(
        dx12Texture, &windowsSecurityAttributes, GENERIC_ALL, name,
        &sharedHandle);

    if (result != S_OK) {

        std::cout << "Error could not create shared handle: " << std::hex << result << "\n";
    }

    std::cout << "Getting d3d allocation info\n";

    D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
    d3d12ResourceAllocationInfo = mpD3D12Device->GetResourceAllocationInfo(0, 1, &resourceDesc);
    size_t actualSize = d3d12ResourceAllocationInfo.SizeInBytes;
    size_t alignment = d3d12ResourceAllocationInfo.Alignment;

    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;

    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = actualSize;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

    std::cout << "Actual memory size: " << externalMemoryHandleDesc.size << "\n";

    cudaError_t e1 = cudaImportExternalMemory(&mCudaExternalMemoryDecoderOutputTexture, &externalMemoryHandleDesc);

    if (e1 != cudaSuccess) {

        std::cout << "Error importing CUDA external memory: " << e1 << "\n";
    }

    cudaExternalMemoryMipmappedArrayDesc mipmapDescriptor;
    memset(&mipmapDescriptor, 0, sizeof(mipmapDescriptor));

    cudaChannelFormatDesc formatDesc;

    formatDesc.f = cudaChannelFormatKindUnsignedNormalized8X4;
    formatDesc.x = 8;
    formatDesc.y = 8;
    formatDesc.z = 8;
    formatDesc.w = 8;

    mipmapDescriptor.extent = make_cudaExtent(1920, 1080, 0);
    mipmapDescriptor.formatDesc = formatDesc;
    mipmapDescriptor.numLevels = 1;
    mipmapDescriptor.flags = cudaArraySurfaceLoadStore;

    cudaError_t e2 = cudaExternalMemoryGetMappedMipmappedArray(&mpCudaDecoderOuputTexture, mCudaExternalMemoryDecoderOutputTexture, &mipmapDescriptor);

    if (e2 != cudaSuccess) {

        std::cout << "Error getting CUDA external mapped mip-mapped array: " << e2 << "\n";
        std::cout << cudaGetErrorString(e2) << "\n";
    }

    std::cout << "Retreiving mip-map array level 0\n";

    cudaArray_t cuArray{};
    cudaError_t e3 = cudaGetMipmappedArrayLevel(&cuArray, mpCudaDecoderOuputTexture, 0);
    cudaExtent arrayExtent;

    cudaArrayGetInfo(nullptr, &arrayExtent, nullptr, cuArray);

    std::cout << "Array is: " << arrayExtent.width << "x" << arrayExtent.height << "\n";

    if (e3 != cudaSuccess) {

        std::cout << "Error getting mipmap array level 0: " << e3 << "\n";
    }

    cudaResourceDesc cuResDesc{};
    cuResDesc.resType = cudaResourceTypeArray;
    cuResDesc.res.array.array = cuArray;

    std::cout << "Making CUDA surface\n";

    cudaError_t e4 = cudaCreateSurfaceObject(&mpCudaDecoderOutputSurface, &cuResDesc);

    if (e4 != cudaSuccess) {

        std::cout << "Error creating surface object: " << e3 << "\n";
    }

    CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr*)&mPDecoderFrame, getDecoderFrameSize()));
    //CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr*)&mPDecoderRGBAFrame, 1920 * 1080 * 4));
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

    std::cout << "Decoding frame buffer \n";

    // packet.flags |= CUVID_PKT_DISCONTINUITY;
    // bitstream_size += mVEncodeOutData.size();
    // std::cout << "mVEncodeOutData.size() " << mVEncodeOutData.size() << "\n";
    NVDEC_API_CALL(cuvidParseVideoData(mHParser, &packet));
}

int EncodeDecode::handleSequenceChange(CUVIDEOFORMAT* pFormat) {


    unsigned int width = pFormat->coded_width;
    unsigned int height = pFormat->coded_height;

    std::cout << "Decoder resolution changed to : " << width << "x" << height << "\n";

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

    // std::cout << "Pic params width: " << pPicParams->PicWidthInMbs * 16 << "\n";
    // std::cout << "Old width: " << mOldWidth << "\n";

    // std::cout << "Begining picture decode\n";

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

    // std::cout << "Mapping video frame\n";

    NVDEC_API_CALL(cuvidMapVideoFrame(mHDecoder, 0, &dpSrcFrame, &nSrcPitch, &videoProcessingParameters));

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
    //cuMemcpy2DAsync(&m, mCuvidStream);
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

    std::cout << "Performing colour conversion\n";

    MyNv12ToColor32(mPDecoderFrame, 1920, mpCudaDecoderOutputSurface, 1920 * 4, 1920, 1080);

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCudaContext)); // push the CUDA context (mCudaContext) onto the current thread's CUDA context stack.
    //  copying data from mPDecoderRGBAFrame (GPU device memory) to mPHostRGBAFrame (CPU host memory)

    //CUDA_DRVAPI_CALL(cuMemcpyDtoH(mPHostRGBAFrame, mPDecoderRGBAFrame, 1920 * 1080 * 4));

    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL)); // o pop the current CUDA context off the CUDA context stack to release resources

    // show mPDecoderRGBAFrame , access cuda memory from directx12 is difficult
    // create directx12 texture, register with cuda xxx, write to this texture
    // don't need to transfer from cuda to cpu and to directx12
    // do: make the memory using directx12, write to the resource

    // pRenderContext->updateTextureData(mDecodedFrame, mPHostRGBAFrame);
    //mDecodedFrame = mPHostRGBAFrame;

    // snprintf(szDecOutFilePath, sizeof(szDecOutFilePath), "%s%d.bmp", decBaseFilePath, fcount);
    // writeBMP(szDecOutFilePath, mDecodedFrame, 1920, 1080);

    //mpRenderContextDecode->updateTextureData(mPDecoderOutputTexture1080.get(), mDecodedFrame);

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


void EncodeDecode::setNNOutputPrefix(std::string filename)
{
    nnOutputFilename = filename;
    fs::path dirPath = fs::path(szNNOutputPrefixFilePath) / filename;

     // Create the file
    std::ofstream file(dirPath);
    if (file.is_open()) {
        std::cout << "Empty CSV file created: " << filename << std::endl;
        file.close();
    } else {
        std::cerr << "Failed to create the file: " << filename << std::endl;
    }
}

void EncodeDecode::setExperimentCSVPrefix(std::string filename)
{
    experimentFilename = filename;
    fs::path dirPath = fs::path(szExperimentPrefixFilePath) / filename;

     // Create the file
    std::ofstream file(dirPath);
    if (file.is_open()) {
        std::cout << "Empty CSV file created: " << filename << std::endl;
        file.close();
    } else {
        std::cerr << "Failed to create the file: " << filename << std::endl;
    }
}


// void EncodeDecode::initDirectML()
// {
//     sessionOptions = new Ort::SessionOptions();
//     sessionOptions->DisableMemPattern();
//     sessionOptions->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
//     sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

//     OrtSessionOptionsAppendExecutionProvider_DML(*sessionOptions, 1);

//     // By passing in an explicitly created DML device & queue, the DML execution provider sends work
//     // to the desired device. If not used, the DML execution provider will create its own device & queue.
//     const OrtApi& ortApi = Ort::GetApi();
//     ortApi.AddFreeDimensionOverrideByName(*sessionOptions, "batch_size", 1);

//     // obtains a pointer to the DML API
//     // ortDmlApi now points to the DML-specific API functions provided by ONNX Runtime
//     Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

//     // Load ONNX model into a session.
//     env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "DirectML_CV");
//     ortSession = new Ort::Session(*env, modelPath.wstring().c_str(), *sessionOptions);
// }

// // load all scenes for experiment
// void EncodeDecode::loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo)
// {

//     mpScenes.push_back(Scene::create(getDevice(), path));
//     // mpScenes.push_back(Scene::create(getDevice(), "suntemple_statue/suntemple_statue04.fbx"));
//     // mpScene = Scene::create(getDevice(), path);
// }

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


void EncodeDecode::loadAllScenes() {
    // preloading takes time

    // max jod vs 0.25
    mpScenes.push_back(Scene::create(getDevice(), "living_room/path2_seg3.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "living_room/path1_seg3.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "living_room/path5_seg2.fbx"));

    // mpScenes.push_back(Scene::create(getDevice(), "crytek_sponza/path3_seg1.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "crytek_sponza/path1_seg2.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "crytek_sponza/path2_seg3.fbx"));

    // mpScenes.push_back(Scene::create(getDevice(), "gallery/path1_seg1.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "gallery/path1_seg2.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "gallery/path1_seg3.fbx"));

    // mpScenes.push_back(Scene::create(getDevice(), "living_room/path1_seg3.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "living_room/path4_seg1.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "living_room/path1_seg1.fbx"));

    // mpScenes.push_back(Scene::create(getDevice(), "lost_empire/path1_seg1.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "lost_empire/path1_seg3.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "lost_empire/path3_seg1.fbx"));

    // mpScenes.push_back(Scene::create(getDevice(), "room/path1_seg1.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "room/path2_seg2.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "room/path3_seg3.fbx"));

    // mpScenes.push_back(Scene::create(getDevice(), "suntemple/path3_seg1.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "suntemple/path5_seg2.fbx"));
    // mpScenes.push_back(Scene::create(getDevice(), "suntemple/path1_seg2.fbx"));
}

void EncodeDecode::setScene(unsigned int index) {

    std::string log;

    // TODO: create a scene
    mpScene = mpScenes.at(index);

    if (mpRenderGraph) {

        mpRenderGraph->setScene(mpScene);
        mpRenderGraph->compile(mpRenderGraph->getDevice()->getRenderContext(), log);
        // mpRenderGraph->getDevice()->getRenderContext()->clearTexture(mpRenderGraph->getOutput("TAA.colorOut")->asTexture().get(), Falcor::float4(0, 0, 0, 0));
        mpRenderGraph->getDevice()->getRenderContext()->clearTexture(mpRenderGraph->getOutput("GBuffer.mvec")->asTexture().get(), Falcor::float4(0, 0, 0, 0));
    }

    // currentSceneIdx = index * 2;
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
    mpCamera->setAspectRatio(16.0f / 9.0f);

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
    targetBitrate = br;
    std::cout << "setBitRate  " << bitRate << "\n";
    std::cout << "target bitrate  " << targetBitrate << "\n";

    if (mHEncoder != nullptr)
    {
        mEncoderInitializeParams.encodeConfig->rcParams.maxBitRate = br * 1000;
        mEncoderInitializeParams.encodeConfig->rcParams.averageBitRate = br * 1000;
        mEncoderInitializeParams.encodeConfig->rcParams.vbvBufferSize = (br * 1000* mEncoderInitializeParams.frameRateDen / mEncoderInitializeParams.frameRateNum) * 5;
        mEncoderInitializeParams.encodeConfig->rcParams.vbvInitialDelay = mEncoderInitializeParams.encodeConfig->rcParams.vbvBufferSize;

        NV_ENC_RECONFIGURE_PARAMS reconfig_params = {0};
        reconfig_params.reInitEncodeParams = mEncoderInitializeParams;
        reconfig_params.resetEncoder = 1;
        reconfig_params.forceIDR = 1;
        reconfig_params.version = NV_ENC_RECONFIGURE_PARAMS_VER;

        // NvEncReconfigureEncoder(mHEncoder, &reconfig_params);
        NVENC_API_CALL(mNVEnc.nvEncReconfigureEncoder(mHEncoder, &reconfig_params));
    }
}



void EncodeDecode::setFrameRate(unsigned int fps)
{

    SampleApp::setTargetFramerate(fps);
    frameRate = fps; // Assign the private member
    // targetFrameTime = 1.0f / frameRate;
    std::cout << "setting frame rate to: " << frameRate << "\n";

    if (mHEncoder != nullptr)
    {
        mEncoderInitializeParams.frameRateNum = fps;
        mEncoderInitializeParams.frameRateDen = 1;
        NV_ENC_RECONFIGURE_PARAMS reconfig_params = {0};
        reconfig_params.reInitEncodeParams = mEncoderInitializeParams;
        reconfig_params.resetEncoder = 0;
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
    std::cout << "mWidth  " << mWidth << " mHeight  " << mHeight << "\n";
    getWindow()->resize(width, height); // Internally resizes the swapchain (back buffer)


    // std::cout << "Setting resolution to: " << width << "x" << height << "\n";

    if (mHEncoder != nullptr)
    {

        // std::cout << "Setting resolution for encoder\n";
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
    }

    mpRtOut = getDevice()->createTexture2D(
        width, height, ResourceFormat::BGRA8Unorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );

    auto fbo = getTargetFbo();
    std::cout << "FBO: " << fbo->getWidth() << "x" << fbo->getHeight() << std::endl;

    if (mpRenderGraph)
    {
        mpRenderGraph->setInput("TAA.colorIn", mpRtOut);
        mpRenderGraph->onResize(fbo.get());
        std::cout << "RenderGraph resized to: " << fbo->getWidth() << "x" << fbo->getHeight() << std::endl;

        mpRenderGraph->compile(mpDevice->getRenderContext());  // REQUIRED!
        // makeEncoderInputBuffers(1); // The input buffer is a GPU texture (ID3D12Resource) that you copy your frame into before encoding.
    }
}


void EncodeDecode::setSpeed(float input)
{
    speed = input;
    std::cout << "\n\n\nsetSpeed  " << speed << "\n";
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


void EncodeDecode::renderRT(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo, uint32_t width, uint32_t height)
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

    // pRenderContext->blit(motionVectorTexture->getSRV(), mpRtMip->getRTV());
    // createMipMaps(pRenderContext);
}


int runMain(int argc, char** argv)
{
    std::cout.setf( std::ios_base::unitbuf );

    // unsigned int bitrate = std::stoi(argv[1]);
    // unsigned int framerate = std::stoi(argv[2]);
    // unsigned int width = std::stoi(argv[3]);
    // unsigned int height = std::stoi(argv[4]);
    // std::string scene = argv[5];
    // unsigned int speedInput = std::stoi(argv[6]);
    // std::string scenePath = argv[7];

    unsigned int width = 1280; // 1920 1280
    unsigned int height = 720; // 1080 720
    unsigned int bitrate = 8000;
    unsigned int framerate = 30;
    std::string scene = "crytek_sponza";
    unsigned int speedInput = 1;
    std::string scenePath = "crytek_sponza/path1_seg1.fbx"; // no texture, objects are black

    std::cout << "\n\nframerate runmain  " << framerate << "\n";
    std::cout << "bitrate runmain  " << bitrate << "\n";
    std::cout << "width runmain  " << width << "\n";
    std::cout << "height runmain  " << height << "\n";
    std::cout << "scene " << scene << std::endl;
    std::cout << "speed " << speedInput << std::endl;
    std::cout << "scenePath " << scenePath << std::endl;

    SampleAppConfig config;
    config.windowDesc.title = "EncodeDecode";
    config.colorFormat = ResourceFormat::BGRA8Unorm;
    config.windowDesc.width = width;
    config.windowDesc.height = height;
    config.windowDesc.resizableWindow = true;

    EncodeDecode encodeDecode(config);
    encodeDecode.setBitRate(bitrate); // 3000 bits per second,  3000 000 bits per second
    encodeDecode.setFrameRate(framerate);
    encodeDecode.setDefaultScene(scenePath);
    encodeDecode.setSpeed(speedInput);

    return encodeDecode.run();
}

int EncodeDecode::run()
{
    return SampleApp::run();
}

int main(int argc, char** argv)
{
    std::cout << "Current Path: " << std::filesystem::current_path() << std::endl;
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
