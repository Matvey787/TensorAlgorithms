#include <gtest/gtest.h>
#include <gmock/gmock.h>

import tensor_gen;
import tensor_io;
import tensor_conv;
import tensor_bench;

namespace 
{

tensor::Tensor<float>

} // namespace


TEST(tensorConvolution, sendWinogradEmptyTensor)
{
    EXPECT_THROW(tensor::conv_winograd(tensor::Tensor{}, tensor::Tensor{}), std::runtime_error())
}

TEST(tensorConvolution, sendWinogradEmptyKernel)
{
    EXPECT_THROW(tensor::conv_winograd(tensor::Tensor{}, tensor::Tensor{}), std::runtime_error())
}

TEST(tensorConvolution, sendWinogradEmptyTensor)
{
    EXPECT_THROW(tensor::conv_winograd(tensor::Tensor{}, tensor::Tensor{}), std::runtime_error())
}

TEST(tensorConvolution, sendWinogradEmptyTensor)
{
    EXPECT_THROW(tensor::conv_winograd(tensor::Tensor{}, tensor::Tensor{}), std::runtime_error())
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
