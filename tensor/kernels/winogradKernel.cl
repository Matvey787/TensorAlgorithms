__constant float B[4][4] = {
    { 1.0f,  0.0f,  0.0f,  0.0f},
    { 0.0f,  1.0f, -1.0f,  1.0f},
    {-1.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f,  0.0f,  0.0f, -1.0f}
};

__constant float BT[4][4] = {
    { 1.0f,  0.0f, -1.0f,  0.0f},
    { 0.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f, -1.0f,  1.0f,  0.0f},
    { 0.0f,  1.0f,  0.0f, -1.0f}
};

__constant float AT[2][4] = {
    {1.0f, 1.0f,  1.0f,  0.0f},
    {0.0f, 1.0f, -1.0f, -1.0f}
};

__constant float A[4][2] = {
    {1.0f,  0.0f},
    {1.0f,  1.0f},
    {1.0f, -1.0f},
    {0.0f, -1.0f}
};

__constant float G[4][3] = {
    {1.0f,  0.0f, 0.0f},
    {0.5f,  0.5f, 0.5f},
    {0.5f, -0.5f, 0.5f},
    {0.0f,  0.0f, 1.0f}
};

__constant float GT[3][4] = {
    {1.0f, 0.5f,  0.5f, 0.0f},
    {0.0f, 0.5f, -0.5f, 0.0f},
    {0.0f, 0.5f,  0.5f, 1.0f}
};

#define TEN_MUL(outT, leftT, rightT, leftWidth, leftHeight, rightHeight) \
    for (int i = 0; i < (leftWidth); ++i)                                \
        for (int j = 0; j < (rightHeight); ++j)                          \
        {                                                                \
            float accumulator = 0.0f;                                    \
            for (int k = 0; k < (leftHeight); ++k)                       \
                accumulator += *(leftT + i * (leftHeight) + k)           \
                               * *(rightT + k * (rightHeight) + j);      \
            *(outT + i * (rightHeight) + j) = accumulator;               \
        }

__kernel void transformKernel(
    __global const float* kernel_data,
    __global       float* transformedLayer,
    const int iCh
)
{
    int chIdx = get_global_id(0);

    if (chIdx >= iCh) return;

    __global const float* kLayer = kernel_data + chIdx * 9;

    float tmp[12];
    float out[16];

    TEN_MUL(tmp, (__constant float*)G, kLayer, 4, 3, 3);
    TEN_MUL(out, tmp, (__constant float*)GT, 4, 3, 4);

    int base = chIdx * 16;

    for (int rawIdx = 0; rawIdx < 4; ++rawIdx)
        for (int colIdx = 0; colIdx < 4; ++colIdx)
            transformedLayer[base + rawIdx * 4 + colIdx] = out[rawIdx * 4 + colIdx];
}

__kernel void winograd_conv(
    __global const float* input,
    __global const float* transformedLayer,
    __global       float* output,
    const int iH, const int iW,
    const int oH, const int oW,
    const int iCh, const int iBatch
)
{
    int tile_x = get_global_id(0);
    int tile_y = get_global_id(1);
    int bIdx   = get_global_id(2);

    int xStartTileIdx = tile_x * 2;
    int yStartTileIdx = tile_y * 2;

    if (xStartTileIdx >= oW || yStartTileIdx >= oH || bIdx >= iBatch) return;

    float accumulatedLayers[16] = {0};

    for (int chIdx = 0; chIdx < iCh; ++chIdx)
    {
        float tile[16];
        for (int ty = 0; ty < 4; ++ty)
            for (int tx = 0; tx < 4; ++tx)
            {
                int iY = yStartTileIdx + ty;
                int iX = xStartTileIdx + tx;

                if (iY < iH && iX < iW)
                    tile[ty * 4 + tx] = input[((bIdx * iCh + chIdx) * iH + iY) * iW + iX];
                else
                    tile[ty * 4 + tx] = 0.0f;
            }

        float tmp[16];
        TEN_MUL(tmp, (__constant float*)BT, tile, 4, 4, 4);

        float transformedInput[16];
        TEN_MUL(transformedInput, tmp, (__constant float*)B, 4, 4, 4);

        int k_base = chIdx * 16;

        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                accumulatedLayers[r * 4 + c] += transformedLayer[k_base + r * 4 + c]
                                              * transformedInput[r * 4 + c];
    }

    float tmp2[8];
    TEN_MUL(tmp2, (__constant float*)AT, accumulatedLayers, 2, 4, 4);

    float result[4];
    TEN_MUL(result, tmp2, (__constant float*)A, 2, 4, 2);

    bool xOutput_maxIdx = (xStartTileIdx + 1) < oW;
    bool yOutput_maxIdx = (yStartTileIdx + 1) < oH;

    int o00 = ((bIdx * oH) + yStartTileIdx                  ) * oW + xStartTileIdx;
    int o01 = ((bIdx * oH) + yStartTileIdx                  ) * oW + xStartTileIdx + xOutput_maxIdx;
    int o10 = ((bIdx * oH) + yStartTileIdx + yOutput_maxIdx ) * oW + xStartTileIdx;
    int o11 = ((bIdx * oH) + yStartTileIdx + yOutput_maxIdx ) * oW + xStartTileIdx + xOutput_maxIdx;

    output[o00] = result[0                 ][0                 ];
    output[o01] = result[0                 ][0 + xOutput_maxIdx];
    output[o10] = result[0 + yOutput_maxIdx][0                 ];
    output[o11] = result[yOutput_maxIdx    ][xOutput_maxIdx    ];
}

