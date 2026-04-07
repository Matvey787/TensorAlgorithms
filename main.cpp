#include <iostream>

import tensor_lib;

int main() try
{
    tensor::Tensor t1{};
}
catch(const std::exception& e)
{
    std::cerr << e.what() << '\n';
}
