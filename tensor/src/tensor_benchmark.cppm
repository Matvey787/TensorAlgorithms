module;

#include <nlohmann/json.hpp>

#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>

#include <fstream>
#include <filesystem>

export module tensor_bench;

import tensor_gen;
import tensor_conv;

namespace tensor
{

export template<typename ValT>
class BenchResult final
{
public:

    size_t input_h, input_w, input_channels, input_batch;
    size_t kernel_h, kernel_w;

    double naive_ms;
    double winograd_ms;
    double speedup;
};

template<typename ValT>
double measure(auto fn, size_t runs)
{
    std::vector<double> times;
    times.reserve(runs);

    for (size_t i = 0; i < runs; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

template<typename ValT>
BenchResult<ValT> benchmark(const Tensor<ValT>& input, const Tensor<ValT>& kernel, size_t runs = 10,
const bool useGpu = false)
{
    // прогрев
    conv_naive(input, kernel);
    conv_winograd(input, kernel);   

    double naive_ms = measure<ValT>([&]{ conv_naive(input, kernel, useGpu); }, runs);
    double winograd_ms = measure<ValT>([&]{ conv_winograd(input, kernel, useGpu); }, runs);

    return {
        .input_h        = input.height(),
        .input_w        = input.width(),
        .input_channels = input.channels(),
        .input_batch    = input.batchSize(),
        .kernel_h       = kernel.height(),
        .kernel_w       = kernel.width(),
        .naive_ms       = naive_ms,
        .winograd_ms    = winograd_ms,
        .speedup        = naive_ms / winograd_ms
    };
}

export template<typename ValT>
std::vector<BenchResult<ValT>> benchmark_all(const std::vector<Tensor<ValT>>& tensors,
                                             size_t runs = 10,
                                             const bool useGpu = false)
{
    const size_t pairs = tensors.size() / 2;
    std::vector<BenchResult<ValT>> benchmarkData(pairs);

    if (useGpu)
    {
        for (size_t i = 0; i < pairs; ++i)
            benchmarkData[i] = benchmark(tensors[i * 2], tensors[i * 2 + 1], runs, useGpu);
    }
    else
    {
        std::vector<std::thread> threads;
        threads.reserve(pairs);

        for (size_t i = 0; i < pairs; ++i)
        {
            threads.emplace_back([&, i]{
                benchmarkData[i] = benchmark(tensors[i * 2], tensors[i * 2 + 1], runs, useGpu);
            });
        }

        for (auto& t : threads)
            t.join();
    }


    return benchmarkData;
}

export template<typename ValT>
void append_benchmark(const std::vector<BenchResult<ValT>>& newData, 
                             const std::string& outputFile)
{
    std::ofstream file(outputFile, std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + outputFile);
    }

    for (const auto& r : newData)
    {
        nlohmann::json j;
        j["input"]  = { {"height", r.input_h}, {"width", r.input_w},
                        {"channels", r.input_channels}, {"batchSize", r.input_batch} };
        j["kernel"] = { {"height", r.kernel_h}, {"width", r.kernel_w} };
        j["naive_ms"]    = r.naive_ms;
        j["winograd_ms"] = r.winograd_ms;
        j["speedup"]     = r.speedup;

        file << j.dump() << '\n';
    }
}

} // namespace tensor
