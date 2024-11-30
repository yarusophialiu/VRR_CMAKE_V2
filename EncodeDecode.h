#pragma once
#pragma warning(disable:4456)

#include <d3d12.h>
#include <cuda.h>
#include <combaseapi.h>
#include <variant>

#include "nvEncodeAPI.h"
#include "nvcuvid.h"
#include "cuviddec.h"
#include "FramePresenterD3D11.h"


#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/RasterPass.h"
#include "d3dx12.h"


#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h> // The DirectML header from the Windows SDK.
// #include <DirectMLX.h>
// #define ORT_MANUAL_INIT
#include "dml_provider_factory.h"
#include "onnxruntime_cxx_api.h"

//#include "FramePresenterD3D11.h"

using namespace Falcor;

#define ALIGN_UP(s, a) (((s) + (a)-1) & ~((a)-1))
#define THROW_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(localHr)) throw localHr;}
#define THROW_IF_ORT_FAILED(exp) \
    { \
        OrtStatus* status = (exp); \
        if (status != nullptr) \
        { \
            throw ConvertOrtStatusToHResult(*status); \
        } \
    }
HRESULT ConvertOrtStatusToHResult(OrtStatus& status)
{
    // std::string errorMessage = ortApi.GetErrorMessage(&status);
    OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
    OrtErrorCode ortErrorCode = ortApi.GetErrorCode(&status);
    ortApi.ReleaseStatus(&status);

    switch (ortErrorCode)
    {
    // POSIX error codes really are inadequate to convey common errors, like even just bad file format :/.
    // Consider using a custom error domain.
    case OrtErrorCode::ORT_OK:               return S_OK;
    case OrtErrorCode::ORT_INVALID_ARGUMENT: return E_INVALIDARG;
    case OrtErrorCode::ORT_NO_SUCHFILE:      return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case OrtErrorCode::ORT_NOT_IMPLEMENTED:  return E_NOTIMPL;
    case OrtErrorCode::ORT_FAIL:
    case OrtErrorCode::ORT_NO_MODEL:
    case OrtErrorCode::ORT_ENGINE_ERROR:
    case OrtErrorCode::ORT_RUNTIME_EXCEPTION:
    case OrtErrorCode::ORT_INVALID_PROTOBUF:
    case OrtErrorCode::ORT_MODEL_LOADED:
    case OrtErrorCode::ORT_INVALID_GRAPH:
    case OrtErrorCode::ORT_EP_FAIL:
    default:                                return E_FAIL;
    }
}


using Microsoft::WRL::ComPtr;



union ScalarUnion
{
    uint64_t u;
    int64_t i;
    double f;
};


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
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
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
    void seNNOutputPrefix(std::string time);



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

    void extract_patch_from_frame(std::vector<uint8_t>& renderedFrameVal, uint32_t frameWidth, uint32_t frameHeight, \
                                  uint32_t startX, uint32_t startY, std::vector<uint8_t>& patchData);
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
    static int CUDAAPI HandleSequenceChangeProc(void* pUserData, CUVIDEOFORMAT* pFormat)
    {
        std::cout << "Set resolution callback\n";
        return ((EncodeDecode*)pUserData)->handleSequenceChange(pFormat);
    }
    int handleSequenceChange(CUVIDEOFORMAT* pFormat);
    int getDecoderFrameSize();

    void initDirectML();
    void CreateCurrentBuffer();
    void CopyTextureIntoCurrentBuffer();
    void CreateFpsBuffer();
    void CreateBitrateBuffer();
    void CreateResolutionBuffer();
    void CreateVelocityBuffer();
    void CreateFloatBuffer(ComPtr<ID3D12Resource> scalarBuffer, D3D12_VERTEX_BUFFER_VIEW scalarBufferView, float scalarValue);
    void UpdateScalarBuffer(ComPtr<ID3D12Resource> scalarBuffer, float newScalarValue);

    ComPtr<ID3D12Resource> CreateD3D12ResourceOfByteSize(
            ID3D12Device* d3dDevice,
            size_t resourceByteSize,
            D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            );
    ComPtr<ID3D12Resource> CreateD3D12ResourceForTensor(
        ID3D12Device* d3dDevice,
        size_t elementByteSize,
        std::span<const int64_t> tensorDimensions
        );


    Ort::Value CreateTensorValueFromExistingD3DResource(
        OrtDmlApi const& ortDmlApi,
        Ort::MemoryInfo const& memoryInformation,
        ID3D12Resource* d3dResource,
        std::span<const int64_t> tensorDimensions,
        ONNXTensorElementDataType elementDataType,
        /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
    );


    std::string GetTensorName(size_t index, Ort::Session const& session, bool isInput);
    bool IsSupportedOnnxTensorElementDataType(ONNXTensorElementDataType dataType);
    char const* NameOfOnnxTensorElementDataType(ONNXTensorElementDataType dataType);
    size_t ByteSizeOfOnnxTensorElementDataType(ONNXTensorElementDataType dataType);
    void GenerateValueSequence(std::span<std::byte> data, ONNXTensorElementDataType dataType);
    void FillIntegerValues(std::span<std::byte> data, ONNXTensorElementDataType dataType, ScalarUnion value);

    Ort::Value CreateTensorValueUsingD3DResource(
        ID3D12Device* d3dDevice,
        OrtDmlApi const& ortDmlApi,
        Ort::MemoryInfo const& memoryInformation,
        std::span<const int64_t> dimensions,
        ONNXTensorElementDataType elementDataType,
        size_t elementByteSize,
        /*out opt*/ ID3D12Resource** d3dResource,
        /*out*/ void** dmlEpResourceWrapper
    );





    void UploadTensorData(
        ID3D12CommandQueue* commandQueue,
        ID3D12CommandAllocator* commandAllocator,
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* destinationResource,
        std::span<const std::byte> sourceData
    );


    // void WriteTensorElementOfDataType(void* data, ONNXTensorElementDataType dataType, T newValue);


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
    ref<Texture> mPDecoderOutputTexture1080;

    // mprtout blit into these encoding texture
    ref<Texture> mPEncoderInputTexture360;
    ref<Texture> mPEncoderInputTexture480;
    ref<Texture> mPEncoderInputTexture720;
    ref<Texture> mPEncoderInputTexture864;
    ref<Texture> mPEncoderInputTexture1080;

    uint8_t mDecodeLock = 0;

    // Microsoft::WRL::ComPtr<IDMLDevice> mpDmlDevice;
    IDMLDevice* mpDmlDevice = nullptr;
    ID3D12CommandQueue* mpD3dQueue;
    // DML execution provider prefers these session options.
public:
    // ONNX runtime
    Ort::SessionOptions* sessionOptions = nullptr;
    Ort::Env* env;
    Ort::Session* ortSession = nullptr;
    const OrtDmlApi* ortDmlApi = nullptr;
    ComPtr<ID3D12Resource> mpPatchBuffer;
    ref<Texture> mpPatchTexture;
    ComPtr<ID3D12GraphicsCommandList> mpCommandList;
    ComPtr<ID3D12CommandAllocator> mpCommandAllocator;
    ComPtr<ID3D12CommandQueue> mpCommandQueue;
    ComPtr<ID3D12PipelineState> mpPipelineState;
    ComPtr<ID3D12Resource> mpFpsBuffer;
    D3D12_VERTEX_BUFFER_VIEW mpFpsBufferView;
    ComPtr<ID3D12Resource> mpBitrateBuffer;
    D3D12_VERTEX_BUFFER_VIEW mpBitrateBufferView;
    ComPtr<ID3D12Resource> mpResBuffer;
    D3D12_VERTEX_BUFFER_VIEW mpResBufferView;
    ComPtr<ID3D12Resource> mpVelocityBuffer;
    D3D12_VERTEX_BUFFER_VIEW mpVelocityBufferView;

    Ort::AllocatorWithDefaultOptions ortAllocator;


    Ort::Value CreateTensorValueFromD3DResource(
        OrtDmlApi const& dmlApi,
        Ort::MemoryInfo const& memoryInformation,
        ID3D12Resource* d3dResource,
        std::vector<int64_t> inputShape,
        ONNXTensorElementDataType elementDataType,
        /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
    );

    uint32_t inputChannels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    // uint32_t inputElementSize;
    // dummy_4channel_novar.onnx   vrr_classification_float32_4channel
    std::filesystem::path modelPath = "C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/smaller_vrr_fp32.onnx"; //smaller_vrr_fp32.onnx";
    // std::vector<int64_t> inputShape;
    // auto outputName;
    // auto outputTypeInfo;
    // auto outputTensorInfo;
    // auto outputShape;
    // auto outputDataType;

    std::map<int, int> reverse_res_map = {{0, 360}, {1, 480}, {2, 720}, {3, 864}, {4, 1080}};
    std::map<int, int> res_map_by_height = {{360, 640}, {480, 854}, {720, 1280}, {864, 1536}, {1080, 1920}};
    std::map<int, int> reverse_fps_map = {{0, 30}, {1, 40}, {2, 50}, {3, 60}, {4, 70}, {5, 80}, {6, 90}, {7, 100}, {8, 110}, {9, 120}};
    std::vector<float> p_res = {0,0,0,0,0};
    std::vector<float> p_fps = {0,0,0,0,0,0,0,0,0,0};
    std::tuple<int, int> shouldChangeSettings(int currentFps, int currentResolution,  std::vector<float>& fps_probabilities,  std::vector<float>& res_probabilities);
    void initializeProbabilities(int currentFps, int currentRes);





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
    uint32_t mOldWidth = 0;
    uint32_t mOldHeight = 0;

    uint32_t patchWidth = 128;
    uint32_t patchHeight = 128;
    int mResolutionChange = 0;

    ref<Scene> mpScene;
    ref<Camera> mpCamera;
    std::unique_ptr<CameraController> mpCamCtrl;

    ref<RasterPass> mpRasterPass;
    ref<RenderGraph> mpRenderGraph;

    /// Compute pass for computing the luminance of an image.
    ref<ComputePass> mpComputeVelocityPass;
    ref<Buffer> mpVelocity; /// Internal buffer for temporary velocity of the patch.

    ref<Program> mpRaytraceProgram;
    ref<RtProgramVars> mpRtVars;
    ref<Texture> mpRtOut;
    ref<Texture> mpRtMip;
    ref<Resource> motionVectorResource;
    ref<Texture> motionVectorTexture;

    // Ort::SessionOptions* sessionOptions;
    // OrtApi* ortApi;
    // const OrtDmlApi* ortDmlApi;



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
    char szNNOutputPrefixFilePath[256] = "nnOutput/";
    char szOutFilePath[256] = "";
    char szRefOutFilePath[256];
    char szDecOutFilePath[256];
    std::string nnOutputFilename = "";

    std::vector<int> frameNumbersCSV;
    std::vector<std::vector<float>> resProbabilitiesCSV;
    std::vector<std::vector<float>> fpsProbabilitiesCSV;
    void appendRowToCsv(int frameNumber, const std::vector<float>& res_probabilities, const std::vector<float>& fps_probabilities);
    void readCsv(const std::string& filename,
             std::vector<int>& frameNumbers,
             std::vector<std::vector<float>>& resProbabilities,
             std::vector<std::vector<float>>& fpsProbabilities);

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
