// import Utils.Color.ColorHelpers;

// cbuffer PerFrameCB
// {
//     uint2 gResolution;
// }

Texture2D<float4> gInputImage;
RWBuffer<float> gOutputVelocity;

static const uint patchWidth = 128;  // Example patch size (8x8)
static const uint patchHeight = 128;

cbuffer PerFrameCB
{
    uint2 gPatchOffset;
};


// only 1 thread per group, each thread comptues velocity for a 128x128 patch
// dispatchThreadId is start x, y of patch
// SV_DispatchThreadID provides a unique ID for each thread in the entire dispatch
// dispatchThreadId: number of kernels on the x, y, z
[numthreads(1, 1, 1)]
void computeVelocity(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    // const float3 color = gInputImage[pixelPos].xyz;
    // gOutputVelocity[pixelPos.x + gResolution.x * pixelPos.y] = color;
    // compute pathch velocity

    float4 average;

    for (int i = 0; i < patchWidth; ++i) {
        for (int j = 0; j < patchHeight; ++j) {
            uint2 pixelPos = uint2(gPatchOffset.x + i, gPatchOffset.y + j);
            float4 colour = gInputImage[pixelPos];
            average += colour;
        }
    }

    average /= float(patchWidth * patchHeight); // should motion vector be the first 2 vectors?
    gOutputVelocity[0] = average.x;
    gOutputVelocity[1] = average.y;
    gOutputVelocity[2] = average.z;
    gOutputVelocity[3] = average.w;
}

    // python average
    // squared_sum = odd_channel ** 2 + even_channel ** 2
    // sqrt_result = torch.sqrt(squared_sum)
    // average = round(sqrt_result.mean().item(), 3)
