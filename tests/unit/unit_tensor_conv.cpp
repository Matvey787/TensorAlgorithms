#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstddef>

import tensor_gen;
import tensor_conv;

namespace 
{

tensor::Tensor<float>
makeTensor(std::size_t height,    std::size_t width, std::size_t channels, 
           std::size_t batchSize,       float value = 1.0)
{
    tensor::Tensor<float> tempTensor{height, width, channels, batchSize};

    for (std::size_t bIdx = 0; bIdx < batchSize; ++bIdx)
        for (std::size_t cIdx = 0; cIdx < channels; ++cIdx)
            for (std::size_t wIdx = 0; wIdx < width; ++wIdx)
                for (std::size_t hIdx = 0; hIdx < height; ++hIdx)
                    tempTensor(bIdx, cIdx)[wIdx, hIdx] = value;

    return tempTensor;
}

constexpr const float EPS = 1e-5f;

} // namespace








// Winograd Validation

TEST(WinogradConvolutionValidation, NonSquareKernel)
{
    auto input  = makeTensor(5, 5, 1, 1);
    auto kernel = makeTensor(3, 5, 1, 1);

    EXPECT_THROW(tensor::conv_winograd(input, kernel), std::runtime_error);
}

TEST(WinogradConvolutionValidation, KernelLargerThanInputTensor)
{
    auto input  = makeTensor(2, 2, 1, 1);
    auto kernel = makeTensor(3, 3, 1, 1);

    EXPECT_THROW(tensor::conv_winograd(input, kernel), std::runtime_error);
}

TEST(WinogradConvolutionValidation, Non3x3KernelThrows)
{
    auto input  = makeTensor(6, 6, 1, 1);
    auto kernel = makeTensor(5, 5, 1, 1);

    EXPECT_THROW(tensor::conv_winograd(input, kernel), std::runtime_error);
}








// Naive Validation

TEST(NaiveConvolutionValidation, KernelLargerThanInputTensor)
{
    auto input  = makeTensor(2, 4, 1, 1);
    auto kernel = makeTensor(4, 4, 1, 1);

    auto input2  = makeTensor(2, 4, 1, 1);
    auto kernel2 = makeTensor(4, 4, 1, 1);
    
    EXPECT_THROW(tensor::conv_naive(input,  kernel),  std::runtime_error);
    EXPECT_THROW(tensor::conv_naive(input2, kernel2), std::runtime_error);
}








// Output Tensor Validation

TEST(OutputShape, NaiveShape)
{
    auto input  = makeTensor(7, 9, 1, 2);
    auto kernel = makeTensor(3, 3, 1, 1);
    auto out    = tensor::conv_naive(input, kernel);

    EXPECT_EQ(out.height(),    5u);
    EXPECT_EQ(out.width(),     7u);
    EXPECT_EQ(out.channels(),  1u);
    EXPECT_EQ(out.batchSize(), 2u);
}

TEST(OutputShape, WinogradShape)
{
    auto input  = makeTensor(7, 9, 1, 2);
    auto kernel = makeTensor(3, 3, 1, 1);
    auto out    = tensor::conv_winograd(input, kernel);

    EXPECT_EQ(out.height(),    5u);
    EXPECT_EQ(out.width(),     7u);
    EXPECT_EQ(out.channels(),  1u);
    EXPECT_EQ(out.batchSize(), 2u);
}

TEST(OutputShape, OddOutputDimension)
{
    auto input  = makeTensor(6, 7, 1, 1);
    auto kernel = makeTensor(3, 3, 1, 1);

    auto naive    = tensor::conv_naive(input, kernel);
    auto winograd = tensor::conv_winograd(input, kernel);

    EXPECT_EQ(naive.height(),    winograd.height());
    EXPECT_EQ(naive.width(),     winograd.width());
}








// Overall Correctness

TEST(OverallCorrectness, NaiveInput3x3Kernel3x3AllVals1)
{
    // 3×3 input, 3×3 kernel, all 1 → single output pixel = 9
    auto input  = makeTensor(3, 3, 1, 1, 1.0f);
    auto kernel = makeTensor(3, 3, 1, 1, 1.0f);
    auto out    = tensor::conv_naive(input, kernel);

    EXPECT_EQ(out.height(), 1);
    EXPECT_EQ(out.width(), 1);

    auto val = out(0, 0)[0, 0];

    EXPECT_NEAR(val, 9.0, EPS);
}

TEST(OverallCorrectness, WinogradInput3x3Kernel3x3AllVals1)
{
    auto input  = makeTensor(3, 3, 2, 1, 1.0f);
    auto kernel = makeTensor(3, 3, 2, 1, 1.0f);
    auto out    = tensor::conv_winograd(input, kernel);

    EXPECT_EQ(out.height(), 1u);
    EXPECT_EQ(out.width(),  1u);

    auto val = out(0, 0)[0, 0];

    EXPECT_NEAR(val, 18.0f, EPS);
}

TEST(OverallCorrectness, MinimalInput4x4)
{
    auto input  = makeTensor(4, 4, 1, 1, 1.0f);
    auto kernel = makeTensor(3, 3, 1, 1, 1.0f);

    auto naive    = tensor::conv_naive(input, kernel);
    auto winograd = tensor::conv_winograd(input, kernel);

    EXPECT_EQ(naive.height(), winograd.height());
    EXPECT_EQ(naive.width(),  winograd.width());

    for (std::size_t y = 0; y < naive.height(); ++y)
        for (std::size_t x = 0; x < naive.width(); ++x)
        {
            auto val = naive(0,0)[x,y];
            auto val2 = winograd(0,0)[x,y];

            EXPECT_NEAR(val, val2, EPS);
        }
            
}

TEST(Correctness, MultiChannelEquivalence)
{
    constexpr std::size_t CH = 4;
    auto input  = makeTensor(6, 8, CH, 1, 2.0f);
    auto kernel = makeTensor(3, 3, CH, 1, 0.5f);

    auto naive    = tensor::conv_naive(input, kernel);
    auto winograd = tensor::conv_winograd(input, kernel);

    for (std::size_t y = 0; y < naive.height(); ++y)
        for (std::size_t x = 0; x < naive.width(); ++x)
        {
            auto val = naive(0,0)[x,y];
            auto val2 = winograd(0,0)[x,y];
            EXPECT_NEAR(val, val2, EPS)
                    << "channel=" << CH << " mismatch at (" << x << "," << y << ")";
        }
                
}





int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
