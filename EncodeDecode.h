#pragma once
#pragma warning(disable:4456)

#include <d3d12.h>
#include <cuda.h>

#include "nvEncodeAPI.h"
#include "nvcuvid.h"
#include "cuviddec.h"
#include "FramePresenterD3D11.h"


#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/RasterPass.h"

#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h> // The DirectML header from the Windows SDK.
// #include <DirectMLX.h>

//#include "FramePresenterD3D11.h"

using namespace Falcor;

#define ALIGN_UP(s, a) (((s) + (a)-1) & ~((a)-1))

struct NvEncInputFrame
{
    void* inputPtr = nullptr;
    uint32_t chromaOffsets[2];
    uint32_t numChromaPlanes;
    uint32_t pitch;
    uint32_t chromaPitch;
    NV_ENC_BUFFER_FORMAT bufferFormat;
    NV_ENC_INPUT_RESOURCE_TYPE resourceType;
};

class EncodeDecode : public SampleApp
{
public:
    EncodeDecode(const SampleAppConfig& config);
    ~EncodeDecode();
    int run() override;
    void onLoad(RenderContext* pRenderContext) override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;

// public:
    float targetFrameTime; // Time per frame in seconds
    uint8_t* mDecodedFrame;
    SampleAppConfig mConfig;

    void setBitRate(unsigned int br);    // Setter for bitRate
    void setFrameRate(unsigned int fps); // Setter for frameRate
    void setResolution(unsigned int width, unsigned int height); // Setter for frameRate
    void setSpeed(unsigned int speed); // Setter for frameRate
    void setSceneName(std::string sceneName); // Setter for frameRate
    void setRefPrefix(std::string scene, unsigned int Input);
    void setMotionPrefix(std::string scene, unsigned int Input, unsigned int framerate, unsigned int bitrate, unsigned int height);
    void setDefaultScene(std::string scenePath);


    // void setMotionFile(std::string name); // Setter for frameRate
// private:
    /*


    @Info
        All of the encoding functionality is specified here.


    */
    // Create defualt encoding params, we will change this function a lot to determine the limits/capabilites of the encoding hardware.
    void makeDefaultEncodingParams(
        NV_ENC_INITIALIZE_PARAMS* pIntializeParams,
        GUID codecGuid,
        GUID presetGuid,
        NV_ENC_TUNING_INFO tuningInfo
    );
    // Initialize the NVEnc API and error out if that fails.
    void initEncoder();
    // Register a resource with NVEnc API.
    NV_ENC_REGISTERED_PTR registerNVEncResource(
        void* pBuffer,
        NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
        int width,
        int height,
        int pitch,
        NV_ENC_BUFFER_FORMAT bufferFormat,
        NV_ENC_BUFFER_USAGE bufferUsage,
        NV_ENC_FENCE_POINT_D3D12* pInputFencePoint = nullptr
    );

    void extract_patch_from_frame(std::vector<uint8_t>& renderedFrameVal, uint32_t frameWidth, uint32_t frameHeight, uint32_t patchWidth, uint32_t patchHeight, std::vector<uint8_t>& patchData);
    // Gte the number of chroma planes based on the encoder pixel format.
    uint32_t getEncoderNumChromaPlanes(const NV_ENC_BUFFER_FORMAT bufferFormat);
    // Get encoder frame size, in bytes.
    int getEncoderFrameSize(int width, int height);
    // Get encoder output buffer size in bytes.
    uint32_t getEncoderOutputBufferSize(int width, int height);
    D3D12_RESOURCE_DESC createEncoderResourceDesc(int width, int height);
    D3D12_RESOURCE_DESC createEncoderOutputResourceDesc(int width, int height);
    // Allocate input buffers in GPU memory.
    void makeEncoderInputBuffers(int32_t numInputBuffers);
    // ALlocate output buffers in GPU memory.
    void makeEncoderOutputBuffers(uint32_t numOutputBuffers);
    // Register input resources for the encoder.
    void registerEncoderInputResources(int width, int height, NV_ENC_BUFFER_FORMAT bufferFormat);
    // Register output resources for the encoder.
    void registerEncoderOutputResources();
    // Map NVEnc resource into memory.
    void mapEncoderResource(uint32_t bufferIndex);
    // Wait for DX12 fence, we do this to ensure syncronisation of GPU resources.
    void cpuWaitForFencePoint(ID3D12Fence* pFence, uint64_t nFenceValue);
    // Wait for NVENC operation to complete and write to completion event at event index.
    void waitForCompletionEvent(int eventIndex);
    // Encode the current frame buffer texture into the video sequence.
    // NVENCSTATUS encodeFrameBuffer(std::ofstream& fpOut);
    NVENCSTATUS encodeFrameBuffer();

    // void EncodeDecode::ReleaseInputBuffers();

    /*


    @Info
        All of the decoding functionality is specified here.


    */
    void makeDefaultDecodingParams(CUVIDDECODECREATEINFO* pInitializeParams);
    void initDecoder();
    void makeDecoderOutputBuffers();
    void decodeFrameBuffer();
    static int CUDAAPI HandlePictureDecodeProc(void* pUserData, CUVIDPICPARAMS* pPicParams)
    {
        // TODO: initialize output frame video somewhere, and add to it everytime decode a frame
        return ((EncodeDecode*)pUserData)->handlePictureDecode(pPicParams);
    }
    int handlePictureDecode(CUVIDPICPARAMS* pPicParams);
    int getDecoderFrameSize();


    // directml
    void initDirectML();

    /*


    @Info
        All of the rendering functionality is specified here.

    */

    /*
    will change once we incoporate motion vector
    */
    void loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo);
    void setPerFrameVars(const Fbo* pTargetFbo);
    void renderRaster(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void renderRT(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo, int fCount, uint32_t width, uint32_t height);
    void createMipMaps(RenderContext* pRenderContext);

    int getResolutionIndex(int resolution);


private:
    // Device* mpDevice;
    ref<Device> mpDevice;

    ID3D12Device* mpD3D12Device; // ID3D12Device is used to create various resources, including buffers, textures...
    // FramePresenterD3D11 presenter;
    /*
    more on IDED12Device
    Resource Creation: ID3D12Device
    Pipeline State Creation: state of the graphics pipeline, including shaders, input layouts, and render targets.
    Command Queue and Command List Management:
        ID3D12Device creates command queues to submit command lists to the GPU for execution.
        It also creates command lists that contain rendering commands.
    Descriptor Management：
        Descriptors are used to represent resources and shader parameters

     The Direct3D 12 device provides support for debugging and profiling through tools like PIX (Performance Investigator for
    Xbox) and other third-party profilers.
    */

    /*


    @Info
        Encoder member vars
        encoder written in c, not cpp
    */
    void* mHEncoder = nullptr; // points to anything, can be dereferenced as any type
    NV_ENCODE_API_FUNCTION_LIST mNVEnc;

    NV_ENC_BUFFER_FORMAT mEBufferFormat;

    NV_ENC_INITIALIZE_PARAMS mEncoderInitializeParams = {};
    NV_ENC_CONFIG mEncodeConfig = {};

    std::vector<NV_ENC_INPUT_RESOURCE_D3D12*> mVInputRsrc;   // frame, store a collection of pointers to objects of type
                                                             // NV_ENC_INPUT_RESOURCE_D3D12
    std::vector<NV_ENC_OUTPUT_RESOURCE_D3D12*> mVOutputRsrc; // bitstream received from encoder

    // fence is a synchronization primitive that controls the
    // execution order of commands submitted to the GPU. Fences play a crucial role in
    // coordinating between the CPU and GPU, particularly in scenarios where the
    // CPU needs to wait for the completion of certain GPU operations.

    // here, fence lets the program pause until decoding is done to a frame
    ID3D12Fence* mpInputFence;
    ID3D12Fence* mpOutputFence;
    // ID3D12Fence* mpDecodeFence;
    Falcor::ref<Falcor::Fence> mpDecodeFence;

    uint64_t mNInputFenceVal;
    uint64_t mNOutputFenceVal;
    uint64_t mNDecodeFenceVal;

    HANDLE mEvent;

    int32_t mNEncoderBuffer = 0;

    std::vector<ID3D12Resource*> mVInputBuffers;
    std::vector<ID3D12Resource*> mVOutputBuffers;

    /*
    mEncoderInputTexture is the texture we are going to encode
    for every frame, get the framebuffer, copy it to the texture, and encode the texture
    */
    ID3D12Resource* mEncoderInputTexture;

    std::vector<NvEncInputFrame> mVInputFrames;
    std::vector<NV_ENC_REGISTERED_PTR> mVRegisteredResources;
    std::vector<NV_ENC_REGISTERED_PTR> mVRegisteredResourcesOutputBuffer;
    std::vector<NV_ENC_REGISTERED_PTR> mVRegisteredResourcesInputBuffer;
    std::vector<NvEncInputFrame> mVReferenceFrames;
    std::vector<NV_ENC_REGISTERED_PTR> mVRegisteredResourcesForReference;
    std::vector<NV_ENC_INPUT_PTR> mVMappedInputBuffers;
    std::vector<NV_ENC_OUTPUT_PTR> mVMappedOutputBuffers;
    std::vector<NV_ENC_INPUT_PTR> mVMappedRefBuffers;

    std::vector<void*> mVPCompletionEvent;

    ID3D12GraphicsCommandList* pGfxCommandList;
    ID3D12CommandQueue* pGfxCommandQueue;

    std::vector<uint8_t> mVEncodeOutData; // container constaining bitstream data

    /*
    decoder cant receive data in gpu's memory, must be cpu's memory
    bitstream received from encoder is in gpu's memory,
    so we need to transform back to cpu
    */

    /*


    @Info
        Decoding member vars


    */

    /*
    iterates over bits, when find enough bits to construct a frame, call the function we specify
    the function gives us the bitstream that constitues the frame and we decode the bitstream into a frame
    have decoded frame in color format that we cant use, so we need to translate that
    from e.g. NV12 encoding to RGBA image

    run cuda kernel on the image and convert the image
    */
    CUdevice mCudaDevice; // represents a GPU device
    /*
    a cuda context is a stateful object that encapsulates a set of resources and settings for CUDA operations on a particular device (GPU)
    it establishes a working environment for CUDA kernels and
    manages resources such as memory allocations, module loading, and stream creation.

    a CUDA context is created on a specific GPU device identified by the corresponding CUdevice.
    */
    CUcontext mCudaContext;
    CUvideoctxlock mCtxLock;
    CUvideoparser mHParser = nullptr;
    CUvideodecoder mHDecoder = nullptr;
    CUstream mCuvidStream = 0;
    uint8_t* mPDecoderFrame = nullptr;
    CUdeviceptr mPDecoderRGBAFrame = 0;
    uint8_t* mPHostRGBAFrame = nullptr;

    FramePresenterD3D11* presenterPtr = nullptr;

    // ID3D12Resource* mPDecoderOutputTexture360;
    ref<Texture> mPDecoderOutputTexture360;
    ref<Texture> mPDecoderOutputTexture480;
    ref<Texture> mPDecoderOutputTexture720;
    ref<Texture> mPDecoderOutputTexture864;
    ref<Texture> mPDecoderOutputTexture1080;

    // mprtout blit into these encoding texture
    ref<Texture> mPEncoderInputTexture360;
    ref<Texture> mPEncoderInputTexture480;
    ref<Texture> mPEncoderInputTexture720;
    ref<Texture> mPEncoderInputTexture864;
    ref<Texture> mPEncoderInputTexture1080;

    uint8_t mDecodeLock = 0;

    Microsoft::WRL::ComPtr<IDMLDevice> mpDmlDevice;
    // IDMLDevice* mpDmlDevice = nullptr;



    /*


    @Info
        Rendering member vars

        mwidth, mheight are the encoded and decoded size
        display size is the size of pop up window
        framebuffer object size is the size we want to change, and can be changed in window.h Desc.width, height
        no need to redo cmake .. after changing the w,h
    */
public:
    uint32_t mWidth;
    uint32_t mHeight;

    ref<Scene> mpScene;
    ref<Camera> mpCamera;
    std::unique_ptr<CameraController> mpCamCtrl;

    ref<RasterPass> mpRasterPass;
    ref<RenderGraph> mpRenderGraph;

    ref<Program> mpRaytraceProgram;
    ref<RtProgramVars> mpRtVars;
    ref<Texture> mpRtOut;
    ref<Texture> mpRtMip;
    ref<Resource> motionVectorResource;
    ref<Texture> motionVectorTexture;



    bool mRayTrace = true;
    bool mUseDOF = false;
    bool outputEncodedFrames = false;   // output as h264 file to C:\Users\15142\new\Falcor\Source\Samples\EncodeDecode\encodedH264
    bool outputDecodedFrames = false;   // output as bmp file
    bool outputReferenceFrames = false; // output Falcor rendered frames as bmp file
    // bool showDecode = true;

    uint32_t mSampleIndex = 0xdeadbeef;
    char kDefaultScene[256] = "";
    char szRefPrefixFilePath[256] = "encodedH264/";
    char szMotionPrefixFilePath[256] = "motion/";
    char szOutFilePath[256] = "";
    char szRefOutFilePath[256];
    char szDecOutFilePath[256];
    // const char* refBaseFilePath = "D:/VRR-frame/room-normal/refOutputBMP/";
    // const char* decBaseFilePath = "D:/VRR-frame/room-normal/decOutputBMP/";

    const char* refBaseFilePath = "refOutputBMP/";
    const char* decBaseFilePath = "decOutputBMP/";
    std::ofstream* fpEncOut;

    // char motionFilePath[256] = "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/decOutputBMP/motion.txt";
    char motionFilePath[256] = ""; // search oss

    //const int frameRate = 30;
    //const int frameLimit = 10 * frameRate / 30; // 206, 516

    signed int frameRate;
    uint32_t frameLimit;
    unsigned int bitRate;
    unsigned int speed;
    std::string sceneName;
    // float targetFrameTime; // Time per frame in seconds
    float lastFrameTime = 0.0f; // Time of the last frame render

    int mipLevels;
    int mipLevelsCompute;

    unsigned int decodeMutex = 0;
    unsigned int numOfFrames = 50;

    RenderContext* mpRenderContextDecode = nullptr;
    // std::chrono::steady_clock::time_point last_send_time = std::chrono::steady_clock::now();


    // const Math::float3& incre = float3(0.00088767 - 0.00192412 - 0.00504681);
};
