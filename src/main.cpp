
#include <stdexcept>
#include <iostream>

import tensor_gen;
import tensor_io;
import tensor_conv;
import tensor_bench;

import parser; 

int main(int argc, char** argv) try
{
    parser::Parser parseObj(argc, argv);

    if (parseObj.hasOption("help"))
    {
        std::cout << parseObj.getOptions().help();

        exit(0);
    }

    std::string sourceFile = parseObj.getOptionVal("source");
    std::string outputFile = parseObj.getOptionVal("output");

    if (sourceFile.empty())
    {
        std::cout << parseObj.getOptions().help() << '\n';
        throw std::runtime_error("No source file provided - no generated.");
    }

    auto&& tensors = tensor::read<float>(sourceFile);

    #ifdef COMPARE

    auto&& benchmarkData = tensor::benchmark_all(tensors, 150);


    if (outputFile.empty())
    {
        tensor::append_benchmark(benchmarkData, "benchmark.json");
    }
    else
    {
        tensor::append_benchmark(benchmarkData, outputFile);
    }

    #else

    std::vector<tensor::Tensor<float>> outputTensors;
    outputTensors.reserve(tensors.size() / 2);

    for (auto it = tensors.begin(); it < std::prev(tensors.end()); it += 2)
    {
        outputTensors.push_back(tensor::conv_winograd(*it, *(it + 1)));
    }

    if (outputFile.empty())
    {
        tensor::dump(outputTensors);
    }
    else
    {
        std::ofstream oFile{outputFile};

        tensor::dump(outputTensors, oFile);
    }

    #endif
}
catch(const std::exception& e)
{
    std::cerr << e.what() << '\n';
}
