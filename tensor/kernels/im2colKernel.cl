// A kernel that transforms an image into a TRANSPOSED columnar image.
// By ‘image’, I mean one of the tensors in the input batch.
// A ‘columnar image’ refers to a matrix whose columns are tiles, 
// onto which the row of a flat kernel is then multiplied matrix-wise.




__kernel void im2col(
    __global const float* input, // input [iB * iC * iH * iW]
    __global       float* col,   // col matrix [oH*oW] x [kH*kW*iC]
    const int iH, const int iW,
    const int kH, const int kW,
    const int oH, const int oW,
    const int imageOffset
)
{   
    const int imageChannelIdx  = get_global_id(0);

    const int imageChannelOffset = imageOffset + imageChannelIdx * iH * iW;

    const int iC = get_global_size(0);

    const int kSize   = kH * kW;
    const int colHeight = oH * oW;
    const int colWidth = kSize * iC;

    const int colRawOffset = imageChannelIdx * kSize;

    for (int oy = 0; oy < oH; ++oy)
    {
        for (int ox = 0; ox < oW; ++ox)
        {
            for (int ky = 0; ky < kH; ++ky)
            {
                for (int kx = 0; kx < kW; ++kx)
                {
                    const int colRawIdx = colRawOffset + ky * kW + kx;

                    const int iy = oy + ky;
                    const int ix = ox + kx;

                    __global float* dst = col + (oy * oW + ox) * colWidth + colRawIdx;

                    *dst = input[imageChannelOffset + iy * iW + ix];
                }
            }
        }
    }
}

