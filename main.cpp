#include <iostream>

import tensor_lib;

int main() try
{
    auto&& tensors = tensor::read<float>("test.json");

    tensor::dump(tensors);


}
catch(const std::exception& e)
{
    std::cerr << e.what() << '\n';
}
