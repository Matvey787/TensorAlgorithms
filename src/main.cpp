#include <stdexcept>
#include <iostream>
#include <ranges>
#include <functional>
#include <fstream>

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

        return 0;
    }

    std::string sourceFile;
    auto sourceFileOption = parseObj.getOptionVal("source");
    if(sourceFileOption)
        sourceFile = sourceFileOption->as<std::string>();



    std::string outputFile;
    auto outputFileOption = parseObj.getOptionVal("output");
    if (outputFileOption)
        outputFile = outputFileOption->as<std::string>();



    if (sourceFile.empty())
    {
        std::cout << parseObj.getOptions().help() << '\n';
        throw std::runtime_error("No source file provided - no generated.");
    }


    [[maybe_unused]] bool useGpu{false};
    auto gpuOption = parseObj.getOptionVal("gpu");
    if (gpuOption) useGpu = gpuOption->as<bool>();

    [[maybe_unused]] bool naiveConv{false};
    auto naiveOption = parseObj.getOptionVal("naive");
    if (naiveOption) naiveConv = naiveOption->as<bool>();

    [[maybe_unused]] bool winogradConv{false};
    auto winogradOption = parseObj.getOptionVal("winograd");
    if (winogradOption) winogradConv = winogradOption->as<bool>();

    [[maybe_unused]] bool im2colConv{false};
    auto im2colOption = parseObj.getOptionVal("im2col");
    if (im2colOption) im2colConv = im2colOption->as<bool>();





    auto&& tensors = tensor::read<float>(sourceFile);

    #ifdef COMPARE

    auto&& benchmarkData = tensor::benchmark_all(tensors, 150, useGpu);


    if (outputFile.empty())
    {
        tensor::append_benchmark(benchmarkData, "benchmark.json");
    }
    else
    {
        tensor::append_benchmark(benchmarkData, outputFile);
    }

    #else

    std::function<tensor::Tensor<float>(tensor::Tensor<float>, 
                                        tensor::Tensor<float>, 
                                        bool                  )> convFunc;

    if (naiveConv)
    {
        convFunc = tensor::conv_naive<float>;
    }
    else if (winogradConv)
    {
        convFunc = tensor::conv_winograd<float>;
    }
    else if (im2colConv)
    {
        convFunc = tensor::conv_im2col<float>;
    }
    else
    {
        std::cout << parseObj.getOptions().help() << '\n';
        throw std::runtime_error("Choose naive/winograd/im2col convolution.");
    }


    std::vector<tensor::Tensor<float>> outputTensors;
    outputTensors.reserve(tensors.size() / 2);

    auto chunked_view = std::views::chunk(tensors, 2);

    for (auto chunk : chunked_view)
    {
        if (chunk.size() < 2) break; 

        auto& input = chunk[0];
        auto& kernel = chunk[1];

        outputTensors.emplace_back(convFunc(input, kernel, useGpu));
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
