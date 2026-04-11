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
                accumulator += leftT[i][k] * rightT[k][j];               \
            outT[i][j] = accumulator;                                    \
        }

__kernel void transformKernel(
    __global const float* kernel_data,
    __global       float* kernel_transformed,
    const int iCh
)
{
    int ch = get_global_id(0);
    if (ch >= iCh) return;

    float kLayer[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            kLayer[r][c] = kernel_data[ch * 9 + r * 3 + c];

    float tmp[4][3];
    TEN_MUL(tmp, G, kLayer, 4, 3, 3);

    float out[4][4];
    TEN_MUL(out, tmp, GT, 4, 3, 4);

    int base = ch * 16;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            kernel_transformed[base + r * 4 + c] = out[r][c];
}

__kernel void winograd_conv(
    __global const float* input,
    __global const float* kernel_transformed,
    __global       float* output,
    const int iH, const int iW,
    const int oH, const int oW,
    const int iCh, const int iBatch
)
{
    int tile_x = get_global_id(0);
    int tile_y = get_global_id(1);
    int b      = get_global_id(2);

    int x = tile_x * 2;
    int y = tile_y * 2;

    if (x >= oW || y >= oH || b >= iBatch) return;

    float accumulate[4][4] = {{0}};

    for (int ch = 0; ch < iCh; ++ch)
    {
        float tile[4][4];
        for (int ty = 0; ty < 4; ++ty)
            for (int tx = 0; tx < 4; ++tx)
            {
                int iy = y + ty;
                int ix = x + tx;
                if (iy < iH && ix < iW)
                    tile[ty][tx] = input[((b * iCh + ch) * iH + iy) * iW + ix];
                else
                    tile[ty][tx] = 0.0f;
            }

        // transformedInput = BT * tile * B  → [4][4]
        float tmp[4][4];
        TEN_MUL(tmp, BT, tile, 4, 4, 4);
        float transformedInput[4][4];
        TEN_MUL(transformedInput, tmp, B, 4, 4, 4);

        int k_base = ch * 16;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                accumulate[r][c] += kernel_transformed[k_base + r * 4 + c]
                                  * transformedInput[r][c];
    }

    float tmp2[2][4];
    TEN_MUL(tmp2, AT, accumulate, 2, 4, 4);
    float result[2][2];
    TEN_MUL(result, tmp2, A, 2, 4, 2);

    for (int ry = 0; ry < 2; ++ry)
        for (int rx = 0; rx < 2; ++rx)
        {
            int oy = y + ry;
            int ox = x + rx;
            if (oy < oH && ox < oW)
                output[((b * oH) + oy) * oW + ox] = result[ry][rx];
        }
}


