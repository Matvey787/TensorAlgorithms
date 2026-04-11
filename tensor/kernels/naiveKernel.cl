#define MAX_K 16

#define TEN_MULWISE(outT, leftT, rightT, height, width, lStride, rStride)      \
    for (int i = 0; i < (height); ++i)                                         \
        for (int j = 0; j < (width); ++j)                                      \
        {                                                                      \
            outT[i][j] = leftT[i * (lStride) + j] * rightT[i * (rStride) + j]; \
        }

#define TEN_SUM(out, ten, height, width)          \
    {                                             \
        float _acc = 0.0f;                        \
        for (int i = 0; i < (width); ++i)         \
            for (int j = 0; j < (height); ++j)    \
                _acc += (ten)[i][j];              \
        (out) = _acc;                             \
    }

__kernel void naive_conv(
    __global const float* input,
    __global const float* kernelT,
    __global       float* output,
    const int iH, const int iW,
    const int kH, const int kW,
    const int oH, const int oW,
    const int iCh, const int iBatch
)
{
    int xoIdx = get_global_id(0);
    int yoIdx = get_global_id(1);
    int bIdx  = get_global_id(2);

    if (xoIdx >= oW || yoIdx >= oH || bIdx >= iBatch) return;

    float accumulator = 0;

    for (int chIdx = 0; chIdx < iCh; ++chIdx)
    {
        int iBatchOffset = bIdx * iCh * iH * iW;
        int iLayerOffset = chIdx * iW * iH;
        int iYOffset     = yoIdx * iW;
        int iXOffset     = xoIdx;

        const float* tile = input + iBatchOffset + iLayerOffset + iYOffset + iXOffset;
        
        int kBatchOffset = chIdx * kH * kW;

        const float* kLayer = kernelT + kBatchOffset;

        float resultTensor[MAX_K][MAX_K];
        TEN_MULWISE(resultTensor, tile, kLayer, kH, kW, iW, kW)

        float layerSumm = 0;
        TEN_SUM(layerSumm, resultTensor, kH, kW)

        accumulator += layerSumm;
    }

    int batchOffset = bIdx * oH * oW;
    int yOffset     = yoIdx * oW;
    int xOffset     = xoIdx;

    *(output + batchOffset + yOffset + xOffset) = accumulator;
}

