module;

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <concepts>

#ifdef WITH_OPENCL

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 300

#ifndef CL_MAKE_VERSION
#define CL_MAKE_VERSION(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))
#endif

#include "opencl.hpp"

#endif

export module tensor_conv;

import tensor_gen;

import opencl;

namespace tensor 
{

template<typename T>
concept mustBeFloat = std::same_as<T, float>;




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

    const std::vector<float> iFlatData{input.data()};
    const std::vector<float> kFlatData{kernel.data()};
    std::vector<float> oFlatData(oH * oW * iB, 0.0f);

    cl::Buffer iBuff(
        executor.context(),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * input.size(),
        const_cast<float*>(iFlatData.data())
    );

    cl::Buffer kBuff(
        executor.context(),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * kernel.size(),
        const_cast<float*>(kFlatData.data())
    );
    
    cl::Buffer oBuff(
        executor.context(),
        CL_MEM_READ_WRITE,
        sizeof(float) * oFlatData.size()
    );

    auto& naive_conv = executor.registerKernel("naive_conv");

    naive_conv.setArg(0,  iBuff);
    naive_conv.setArg(1,  kBuff);
    naive_conv.setArg(2,  oBuff);
    naive_conv.setArg(3,  static_cast<int>(iH));
    naive_conv.setArg(4,  static_cast<int>(iW));
    naive_conv.setArg(5,  static_cast<int>(kH));
    naive_conv.setArg(6,  static_cast<int>(kW));
    naive_conv.setArg(7,  static_cast<int>(oH));
    naive_conv.setArg(8,  static_cast<int>(oW));
    naive_conv.setArg(9,  static_cast<int>(iC));
    naive_conv.setArg(10, static_cast<int>(iB));

    executor.queue().enqueueNDRangeKernel(
        naive_conv,
        cl::NullRange,
        cl::NDRange(oW, oH, iB),
        cl::NullRange
    );

    executor.queue().enqueueReadBuffer(
        oBuff,
        CL_TRUE,
        0,
        sizeof(float) * oFlatData.size(),
        oFlatData.data()
    );

    executor.finishQueue();

    return Tensor<float>(std::move(oFlatData), oH, oW, 1, iB);
}

#endif









































































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

    const std::vector<float> iFlatData{input.data()};
    const std::vector<float> kFlatData{kernel.data()};
    std::vector<float> oFlatData(oH * oW * iB, 0.0f);

    cl::Buffer iBuff(
        executor.context(),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * iFlatData.size(),
        const_cast<float*>(iFlatData.data())
    );

    cl::Buffer kBuff(
        executor.context(),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * kernel.size(),
        const_cast<float*>(kFlatData.data())
    );

    cl::Buffer oBuff(
        executor.context(),
        CL_MEM_WRITE_ONLY,
        sizeof(float) * oH * oW * iB
    );

    cl::Buffer transformedKernelBuff(
        executor.context(),
        CL_MEM_READ_WRITE,
        sizeof(float) * kernel.size()
    );

    auto& transformKernel = executor.registerKernel("transformKernel");

    transformKernel.setArg(0, kBuff);
    transformKernel.setArg(1, transformedKernelBuff);
    transformKernel.setArg(2, static_cast<int>(iC));

    executor.queue().enqueueNDRangeKernel(
        transformKernel,
        cl::NullRange,
        cl::NDRange(iC),
        cl::NullRange
    );

    auto& winograd_conv = executor.registerKernel("winograd_conv");

    winograd_conv.setArg(0, iBuff);
    winograd_conv.setArg(1, transformedKernelBuff);
    winograd_conv.setArg(2, oBuff);
    winograd_conv.setArg(3, static_cast<int>(iH));
    winograd_conv.setArg(4, static_cast<int>(iW));
    winograd_conv.setArg(5, static_cast<int>(oH));
    winograd_conv.setArg(6, static_cast<int>(oW));
    winograd_conv.setArg(7, static_cast<int>(iC));
    winograd_conv.setArg(8, static_cast<int>(iB));

    executor.queue().enqueueNDRangeKernel(
        winograd_conv,
        cl::NullRange,
        cl::NDRange((oW + 1) / 2, (oH + 1) / 2, iB),
        cl::NullRange
    );

    executor.queue().enqueueReadBuffer(
        oBuff,
        CL_TRUE,
        0,
        sizeof(float) * oFlatData.size(),
        oFlatData.data()
    );

    executor.finishQueue();

    return Tensor<float>(std::move(oFlatData), oH, oW, 1, iB);
}

#endif




} // namespace tensor