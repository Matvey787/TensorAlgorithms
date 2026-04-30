module;

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <concepts>
#include <numeric>
#include <immintrin.h>

#ifdef WITH_OPENCL

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 300

#ifndef CL_MAKE_VERSION
#define CL_MAKE_VERSION(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))
#endif

#include "opencl.hpp"
#include <clblast.h>

#endif

export module tensor_conv;

import tensor_gen;

import opencl;

namespace tensor 
{

template<typename T>
concept mustBeFloat = std::same_as<T, float>;










// ------------------------------------------------------------------
// Naive Implementation (CPU + GPU)
// ------------------------------------------------------------------










template<typename ValT>
Tensor<ValT> conv_naive_cpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel);

template<typename ValT>
    requires mustBeFloat<ValT>
Tensor<ValT> conv_naive_gpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel);

export template<typename ValT>
Tensor<ValT> conv_naive(const Tensor<ValT>&   input, 
                           const Tensor<ValT>&   kernel, 
                           [[maybe_unused]] bool useGpu = false)
{

    #ifdef WITH_OPENCL

    if (useGpu) return conv_naive_gpu(input, kernel);

    #endif

    return conv_naive_cpu(input, kernel);
}

template<typename ValT>
Tensor<ValT> conv_naive_cpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel)
{
    const std::size_t iH     = input.height();
    const std::size_t iW     = input.width();
    const std::size_t iBatch = input.batchSize();


    const std::size_t kCh    = kernel.channels();
    const std::size_t kH     = kernel.height();
    const std::size_t kW     = kernel.width();

    if (iH < kH || iW < kW)
        throw std::runtime_error("The input tensor cannot be smaller than the kernel.");

    const std::size_t oH = iH - kH + 1;
    const std::size_t oW = iW - kW + 1;

    Tensor<ValT> output(oH, oW, 1, iBatch);

    for (std::size_t b = 0; b < iBatch; ++b)
    {
        for (std::size_t y = 0; y < oH; ++y)
        {
            for (std::size_t x = 0; x < oW; ++x)
            {
                Layer<ValT> accumulateLayer(kH, kW);

                for (std::size_t channel_idx = 0; channel_idx < kCh; ++channel_idx)
                {
                    const auto& iLayer = input(b, channel_idx);
                    const auto& kLayer = kernel(0, channel_idx);


                    Layer<ValT> tile(kH, kW);

                    for (size_t tile_y = 0; tile_y < kH; ++tile_y)
                    {
                        for (size_t tile_x = 0; tile_x < kW; ++tile_x)
                        {
                            if (y + tile_y < iH && x + tile_x < iW)
                                tile[tile_x, tile_y] = iLayer[x + tile_x, y + tile_y];
                            else
                                tile[tile_x, tile_y] = ValT{0}; // Zero padding for boundaries
                        }
                    }

                    accumulateLayer = std::move(accumulateLayer.add(tile.mul_elementWise(kLayer)));
                }
                output(b, 0)[x, y] = accumulateLayer.sum();
            }
        }
    }

    return output;
}


#ifdef WITH_OPENCL

template<typename ValT>
    requires mustBeFloat<ValT>
Tensor<ValT> conv_naive_gpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel)
{
    static opencl::KernelExecutor executor("tensor/kernels/naiveKernel.cl");

    const size_t iH = input.height();
    const size_t iW = input.width();
    const size_t iC = input.channels();
    const size_t iB = input.batchSize();

    const size_t kH = kernel.height();
    const size_t kW = kernel.width();

    const size_t oH = iH - kH + 1;
    const size_t oW = iW - kW + 1;

    if (iH < kH || iW < kW)
        throw std::runtime_error("The input tensor cannot be smaller than the kernel.");

    const std::vector<float> iFlatData = input.data();
    const std::vector<float> kFlatData = kernel.data();

    std::vector<float> oFlatData(oH * oW * iB, 0.0f);

    executor.updateBuffer("iBuff", CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, iFlatData);
    executor.updateBuffer("kBuff", CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kFlatData);
    executor.updateBuffer("oBuff", CL_MEM_WRITE_ONLY, oH * oW * iB);

    auto& naive_conv = executor.registerKernel("naive_conv");

    naive_conv.setArg(0,  executor.getClBuffer("iBuff"));
    naive_conv.setArg(1,  executor.getClBuffer("kBuff"));
    naive_conv.setArg(2,  executor.getClBuffer("oBuff"));
    naive_conv.setArg(3,  static_cast<int>(iH));
    naive_conv.setArg(4,  static_cast<int>(iW));
    naive_conv.setArg(5,  static_cast<int>(kH));
    naive_conv.setArg(6,  static_cast<int>(kW));
    naive_conv.setArg(7,  static_cast<int>(oH));
    naive_conv.setArg(8,  static_cast<int>(oW));
    naive_conv.setArg(9,  static_cast<int>(iC));
    naive_conv.setArg(10, static_cast<int>(iB));

    executor.enqueueNDRange(naive_conv, cl::NDRange(oW, oH, iB));

    executor.fetchBuffer("oBuff", oFlatData);

    executor.finish();

    return Tensor<float>(std::move(oFlatData), oH, oW, 1, iB);
}

#endif










// ------------------------------------------------------------------
// Winograd Implementation (CPU + GPU)
// ------------------------------------------------------------------










template<typename ValT>
Tensor<ValT> conv_winograd_cpu(const Tensor<ValT>& input,
                               const Tensor<ValT>& kernel);

template<typename ValT>
Tensor<ValT> conv_winograd_gpu(const Tensor<ValT>& input,
                               const Tensor<ValT>& kernel);


export template<typename ValT>
Tensor<ValT> conv_winograd(const Tensor<ValT>&   input, 
                           const Tensor<ValT>&   kernel, 
                           [[maybe_unused]] bool useGpu = false)
{

    #ifdef WITH_OPENCL

    if (useGpu) return conv_winograd_gpu(input, kernel);

    #endif

    return conv_winograd_cpu(input, kernel);
}

template<typename ValT>
Tensor<ValT> conv_winograd_cpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel)
{
    const std::size_t iH = input.height();
    const std::size_t iW = input.width();
    const std::size_t iCh = input.channels();
    const std::size_t iBatch = input.batchSize();

    const std::size_t kH = kernel.height();
    const std::size_t kW = kernel.width(); 

    if (kH != 3 || kW != 3)
        throw std::runtime_error("Kernel should be 3x3 for Winograd F(2,3).");

    if (iH < 3 || iW < 3)
        throw std::runtime_error("Minimum size of input matrix for Winograd is 3x3.");

    if (iH < kH || iW < kW)
        throw std::runtime_error("The input tensor cannot be smaller than the kernel.");

    const std::size_t oH = iH - 2;
    const std::size_t oW = iW - 2;
    
    Tensor<ValT> output(oH, oW, 1, iBatch);

    static const Layer<ValT> G = {
        {1.0,  0.0, 0.0},
        {0.5,  0.5, 0.5},
        {0.5, -0.5, 0.5},
        {0.0,  0.0, 1.0}
    };
    static const Layer<ValT> GT = G.transpose();

    static const Layer<ValT> BT = {
        {1.0,  0.0, -1.0,  0.0},
        {0.0,  1.0,  1.0,  0.0},
        {0.0, -1.0,  1.0,  0.0},
        {0.0,  1.0,  0.0, -1.0}
    };
    static const Layer<ValT> B = BT.transpose();

    static const Layer<ValT> AT = {
        {1.0, 1.0,  1.0,  0.0},
        {0.0, 1.0, -1.0, -1.0}
    };
    static const Layer<ValT> A = AT.transpose();


    // generate new layers of kernel
    std::vector<Layer<ValT>> transformedKernel;

    transformedKernel.reserve(iCh);

    for (size_t channel_idx = 0; channel_idx < iCh; ++channel_idx)
    {
        const Layer<ValT>& kLayer = kernel(0, channel_idx);

        Layer<ValT> transformedKernelLayer = G.mul_matrix(kLayer).mul_matrix(GT);

        transformedKernel.push_back(std::move(transformedKernelLayer));
    }


    // main loop
    for (size_t b = 0; b < iBatch; ++b)
    {
        for (size_t y = 0; y < oH; y += 2)
        {
            for (size_t x = 0; x < oW; x += 2)
            {
                Layer<ValT> accumulateLayer(4, 4);

                for (size_t channel_idx = 0; channel_idx < iCh; ++channel_idx)
                {
                    // get convertible tile 4 x 4
                    Layer<ValT> tile(4, 4);
                    for (size_t tile_y = 0; tile_y < 4; ++tile_y)
                    {
                        for (size_t tile_x = 0; tile_x < 4; ++tile_x)
                        {
                            if (y + tile_y < iH && x + tile_x < iW)
                                tile[tile_x, tile_y] = input(b, channel_idx)[x + tile_x, y + tile_y];
                            else
                                tile[tile_x, tile_y] = ValT{0}; // Zero padding for boundaries
                        }
                    }

                    Layer<ValT> transformedInput = BT.mul_matrix(tile).mul_matrix(B);

                    Layer<ValT> M = transformedKernel[channel_idx].mul_elementWise(transformedInput);
                    
                    accumulateLayer = std::move(accumulateLayer.add(M));
                }

                // final transformation
                Layer<ValT> result = AT.mul_matrix(accumulateLayer).mul_matrix(A);

                bool xOutput_maxIdx = (x + 1) < oW;
                bool yOutput_maxIdx = (y + 1) < oH;

                output(b, 0)[x + 0,              y + 0             ] = result[0,              0             ];
                output(b, 0)[x + xOutput_maxIdx, y + 0             ] = result[xOutput_maxIdx, 0             ];
                output(b, 0)[x + 0,              y + yOutput_maxIdx] = result[0,              yOutput_maxIdx];
                output(b, 0)[x + xOutput_maxIdx, y + yOutput_maxIdx] = result[xOutput_maxIdx, yOutput_maxIdx];

            }
        }
    }

    return output;
}



#ifdef WITH_OPENCL
template<typename ValT>
    requires mustBeFloat<ValT>
Tensor<ValT> conv_winograd_gpu(const Tensor<ValT>& input,
                               const Tensor<ValT>& kernel)
{
    static opencl::KernelExecutor executor("tensor/kernels/winogradKernel.cl");

    const size_t iH = input.height();
    const size_t iW = input.width();
    const size_t iC = input.channels();
    const size_t iB = input.batchSize();

    const size_t oH = iH - 2;
    const size_t oW = iW - 2;

    const std::size_t kH = kernel.height();
    const std::size_t kW = kernel.width(); 

    if (kH != 3 || kW != 3)
        throw std::runtime_error("Kernel should be 3x3 for Winograd F(2,3).");

    if (iH < 3 || iW < 3)
        throw std::runtime_error("Minimum size of input matrix for Winograd is 3x3.");

    if (iH < kH || iW < kW)
        throw std::runtime_error("The input tensor cannot be smaller than the kernel.");

    const std::vector<float> iFlatData{input.data()};
    const std::vector<float> kFlatData{kernel.data()};
    std::vector<float> oFlatData(oH * oW * iB, 0.0f);

    executor.updateBuffer("iBuff", CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std::move(iFlatData));
    executor.updateBuffer("kBuff", CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std::move(kFlatData));
    executor.updateBuffer("oBuff", CL_MEM_WRITE_ONLY, oH * oW * iB);
    executor.updateBuffer("transformedKernelBuff", CL_MEM_READ_WRITE, kernel.size());

    auto& transformKernel = executor.registerKernel("transformKernel");

    transformKernel.setArg(0, executor.getClBuffer("kBuff"));
    transformKernel.setArg(1, executor.getClBuffer("transformedKernelBuff"));
    transformKernel.setArg(2, static_cast<int>(iC));

    executor.enqueueNDRange(transformKernel, cl::NDRange(iC));

    auto& winograd_conv = executor.registerKernel("winograd_conv");

    winograd_conv.setArg(0, executor.getClBuffer("iBuff"));
    winograd_conv.setArg(1, executor.getClBuffer("transformedKernelBuff"));
    winograd_conv.setArg(2, executor.getClBuffer("oBuff"));
    winograd_conv.setArg(3, static_cast<int>(iH));
    winograd_conv.setArg(4, static_cast<int>(iW));  
    winograd_conv.setArg(5, static_cast<int>(oH));
    winograd_conv.setArg(6, static_cast<int>(oW));
    winograd_conv.setArg(7, static_cast<int>(iC));
    winograd_conv.setArg(8, static_cast<int>(iB));

    executor.enqueueNDRange(winograd_conv, cl::NDRange((oW + 1) / 2, (oH + 1) / 2, iB));

    executor.fetchBuffer("oBuff", oFlatData);

    executor.finish();

    return Tensor<float>(std::move(oFlatData), oH, oW, 1, iB);
}

#endif


// ------------------------------------------------------------------
// Im2Col Implementation (CPU + GPU)
// ------------------------------------------------------------------




template<typename ValT>
    requires mustBeFloat<ValT>
Tensor<ValT> conv_im2col_cpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel);

template<typename ValT>
    requires mustBeFloat<ValT>
Tensor<ValT> conv_im2col_gpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel);

export template<typename ValT>
Tensor<ValT> conv_im2col(const Tensor<ValT>&   input, 
                           const Tensor<ValT>&   kernel, 
                           [[maybe_unused]] bool useGpu = false)
{

    #ifdef WITH_OPENCL

    if (useGpu) return conv_im2col_gpu(input, kernel);

    #endif

    return conv_im2col_cpu(input, kernel);
}


template<typename ValT>
    requires mustBeFloat<ValT>
__attribute__((target("avx")))
Tensor<ValT> conv_im2col_cpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel)
{
    const std::size_t iH     = input.height();
    const std::size_t iW     = input.width();
    const std::size_t iCh    = input.channels();
    const std::size_t iBatch = input.batchSize();

    const std::size_t kH = kernel.height();
    const std::size_t kW = kernel.width();

    const std::size_t oH = iH - kH + 1;
    const std::size_t oW = iW - kW + 1;

    const std::size_t tileSize   = kH * kW;
    const std::size_t paddedSize = ((tileSize + 7) / 8) * 8;


    ValT* inputTile = static_cast<ValT*>(
        std::aligned_alloc(32, paddedSize * sizeof(ValT)));

    if (!inputTile) throw std::bad_alloc{};

    std::fill(inputTile, inputTile + paddedSize, ValT{0});

    ValT* rawKernelData = static_cast<ValT*>(
        std::aligned_alloc(32, iCh * paddedSize * sizeof(ValT)));

    if (!rawKernelData)
    {
        std::free(inputTile);
        throw std::bad_alloc{};
    }
    std::fill(rawKernelData, rawKernelData + iCh * paddedSize, ValT{0});

    auto kernelRow = [&](std::size_t c) -> ValT*
    {
        return rawKernelData + c * paddedSize;
    };

    for (std::size_t c = 0; c < iCh; ++c)
    {
        const auto& srcChannel = kernel(0, c).data();
        std::copy(srcChannel.begin(), srcChannel.end(), kernelRow(c));
    }


    Tensor<ValT> output(oH, oW, 1, iBatch);

    alignas(32) ValT simdRes[8];

    for (std::size_t b = 0; b < iBatch; ++b)
    {
        for (std::size_t y = 0; y < oH; ++y)
        {
            for (std::size_t x = 0; x < oW; ++x)
            {
                ValT accumulated{0};

                for (std::size_t c = 0; c < iCh; ++c)
                {
                    const tensor::Layer<ValT>& iLayer = input(b, c);

                    for (std::size_t tile_y = 0; tile_y < kH; ++tile_y)
                        for (std::size_t tile_x = 0; tile_x < kW; ++tile_x)
                            inputTile[tile_y * kW + tile_x] = iLayer[x + tile_x, y + tile_y];

                    const ValT* kRow = kernelRow(c);
                    for (std::size_t i = 0; i < paddedSize; i += 8)
                    {
                        __m256 iv = _mm256_load_ps(inputTile + i);
                        __m256 kv = _mm256_load_ps(kRow + i);
                        __m256 rv = _mm256_mul_ps(iv, kv);
                        _mm256_store_ps(simdRes, rv);

                        accumulated += simdRes[0] + simdRes[1] + simdRes[2] + simdRes[3]
                                     + simdRes[4] + simdRes[5] + simdRes[6] + simdRes[7];
                    }
                }

                output(b, 0)[x, y] = accumulated;
            }
        }
    }

    std::free(inputTile);
    std::free(rawKernelData);

    return output;
}


#ifdef  WITH_OPENCL
template<typename ValT>
    requires mustBeFloat<ValT>
Tensor<ValT> conv_im2col_gpu(const Tensor<ValT>& input, const Tensor<ValT>& kernel)
{
    const size_t iH = input.height();
    const size_t iW = input.width();
    const size_t iC = input.channels();
    const size_t iB = input.batchSize();

    const size_t kH = kernel.height();
    const size_t kW = kernel.width();
    const size_t kC = kernel.channels();

    if (iH < kH || iW < kW)
        throw std::runtime_error("The input tensor cannot be smaller than the kernel.");
        
    if (iC != kC)
        throw std::runtime_error("Kernel channels must match input channels.");

    const size_t oH = iH - kH + 1;
    const size_t oW = iW - kW + 1;
    const size_t kSize = kH * kW;
    const size_t colRows = oH * oW;
    const size_t colCols = kSize * iC;

    static opencl::KernelExecutor executor("tensor/kernels/im2colKernel.cl");

    const std::vector<float> iFlatData = input.data();
    const std::vector<float> kFlatData = kernel.data();
    
    executor.updateBuffer("iBuff", CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, iFlatData);
    executor.updateBuffer("kBuff", CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kFlatData);
    
    executor.updateBuffer("colBuff", CL_MEM_READ_WRITE, colRows * colCols);
    
    executor.updateBuffer("oBuff", CL_MEM_READ_WRITE, colRows);

    auto& im2colKernel = executor.registerKernel("im2col");

    Tensor<ValT> output(oH, oW, 1, iB);

    for (size_t b = 0; b < iB; ++b)
    {
        const int imageOffset = static_cast<int>(b * iC * iH * iW);
        
        im2colKernel.setArg(0, executor.getClBuffer("iBuff"));
        im2colKernel.setArg(1, executor.getClBuffer("colBuff"));
        im2colKernel.setArg(2, static_cast<int>(iH));
        im2colKernel.setArg(3, static_cast<int>(iW));
        im2colKernel.setArg(4, static_cast<int>(kH));
        im2colKernel.setArg(5, static_cast<int>(kW));
        im2colKernel.setArg(6, static_cast<int>(oH));
        im2colKernel.setArg(7, static_cast<int>(oW));
        im2colKernel.setArg(8, imageOffset);

        executor.enqueueNDRange(im2colKernel, cl::NDRange(iC));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Now kFlatData is row matrix [1 x kernel size]
        // colBuff is transposed column matrix, 
        // which will need to be transposed for matrix multiplication

        
        /*
        For example:

        input = [[1, 2,  3,  4,  |
                  5, 6,  7,  8,  | first layer
                  9, 10, 11, 12],|
                 [1, 2,  3,  4,    |
                  5, 6,  7,  8,    | second layer
                  9, 10, 11, 12]]  |

        kernel = [[9, 8, 7,
                   6, 5, 4,
                   3, 2, 1],
                  [1, 2, 3
                   4, 5, 6,
                   7, 8, 9]]
        
        kFlatData = [9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        colBuff = [
                    [1, 2, 3, 5, 6, 7, 9, 10, 11, 1, 2, 3, 5, 6, 7, 9, 10, 11],
                    [2, 3, 4, 6, 7, 8, 10, 11, 12, 2, 3, 4, 6, 7, 8, 10, 11, 12]
                  ]
        
        output = kFlatData * transposed colBuff

        */


        clblast::StatusCode status = clblast::Gemm<float>(
            clblast::Layout::kRowMajor, // row matrixes
            clblast::Transpose::kNo,    // no need to transpose kBuff
            clblast::Transpose::kYes,   // need to transpose colBuff (see explanation before)
            1,                          // number of rows in kBuff
            colRows,                    // width of transposed kBuff
            colCols,                    // width of kBuff = height of transposed colBuff = colCols
            alpha,                      // alpha * kBuff * colBuff, so alpha = 1
            executor.getClBuffer("kBuff")(),  0, colCols, // colCols - row length step in kernel 
            executor.getClBuffer("colBuff")(), 0, colCols, // colCols - row length in initial
            beta, // beta = 0
            executor.getClBuffer("oBuff")(),  0, colRows, // colRows - row length in oBuff
            &(executor.queue()()), nullptr
        );

        if (status != clblast::StatusCode::kSuccess)
        {
            throw std::runtime_error(
                "clBLAST Gemm failed at batch " + std::to_string(b) +
                ", error code: " + std::to_string(static_cast<int>(status))
            );
        }

        std::vector<float> batchOutData(colRows);

        executor.fetchBuffer("oBuff", batchOutData);

        Layer<ValT> layer(batchOutData.begin(), batchOutData.end(), oH);
        
        output(b, 0) = std::move(layer);
    }

    return output;
}

#endif





} // namespace tensor