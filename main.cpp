#include <iostream>

import tensor_gen;
import tensor_io;
import tensor_conv;

 
int main() try
{
    auto&& tensors = tensor::read<float>("test3.json");

    auto&& input = tensors[0];

    auto&& kernel = tensors[1];

    std::cout << "Input matrix: \n";
    tensor::dump(input);

    std::cout << "\n\nKernel tensor:\n";
    tensor::dump(kernel);

    tensor::Tensor<float> ans1 { tensor::conv_naive(input, kernel) };

    tensor::Tensor<float> ans2{ tensor::conv_winograd(input, kernel) };


    std::cout << "Naive answer: \n";

    tensor::dump(ans1);

    std::cout << "\n\nWinograd answer: \n";

    tensor::dump(ans2);


}
catch(const std::exception& e)
{
    std::cerr << e.what() << '\n';
}
