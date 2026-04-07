module;

#include <boost/json.hpp>

#include <exception>
#include <concepts>
#include <vector>
#include <stdexcept>

export module tensor_lib;

namespace tensor
{

export template<typename ValT = float>
class Tensor final
{
    size_t height_{0};
    size_t width_{0};
    size_t channels_{0};
    size_t batchSize_{0};

    class layer final
    {
        size_t height_;
        size_t width_;
        std::vector<ValT> data_;

    public:
        layer(size_t h, size_t w)
            : height_(h), width_(w), data_(h * w, ValT{})
        {}

        [[gnu::always_inline]]
        ValT& operator[](size_t x_idx, size_t y_idx)
        {
            if (x_idx >= width_ || y_idx >= height_) [[unlikely]]
                throw std::runtime_error("Layer indexes out of bounds.");

            return data_[width_ * y_idx + x_idx];
        }

        [[gnu::always_inline]]
        const ValT& operator[](size_t x_idx, size_t y_idx) const
        {
            if (x_idx >= width_ || y_idx >= height_) [[unlikely]]
                throw std::runtime_error("Layer indexes out of bounds.");

            return data_[width_ * y_idx + x_idx];
        }

        [[nodiscard]] size_t height() const noexcept { return height_; }
        [[nodiscard]] size_t width()  const noexcept { return width_;  }
    };

    std::vector<layer> layers_;

public:
    Tensor() = default;

    Tensor(size_t h, size_t w, size_t c, size_t batch)
        : height_(h), width_(w), channels_(c), batchSize_(batch)
    {
        layers_.reserve(c * batch);
        for (size_t i = 0; i < c * batch; ++i)
            layers_.emplace_back(h, w);
    }

    [[nodiscard]] size_t height()    const noexcept { return height_;    }
    [[nodiscard]] size_t width()     const noexcept { return width_;     }
    [[nodiscard]] size_t channels()  const noexcept { return channels_;  }
    [[nodiscard]] size_t batchSize() const noexcept { return batchSize_; }

    const std::vector<layer>& getLayers() const 
    {
        return layers_;
    }

    layer& operator()(size_t batch_idx, size_t channel_idx)
    {
        return layers_[batch_idx * channels_ + channel_idx];
    }

    const layer& operator()(size_t batch_idx, size_t channel_idx) const
    {
        return layers_[batch_idx * channels_ + channel_idx];
    }

};




export template<typename ValT>
Tensor<ValT> conv_naive(const Tensor<ValT>& input, const Tensor<ValT>& kernel)
{
    const size_t iH = input.height();
    const size_t iW = input.width();
    const size_t iBatchsize = input.batchSize();

    const size_t kH = kernel.height();
    const size_t kW = kernel.width();

    const size_t iCh = input.channels();
    const size_t kCh = kernel.channels();

    if (kH > iH || kW > iW) [[unlikely]] throw std::runtime_error("Kernel larger than input.");

    if (iCh != kCh) [[unlikely]]
        throw std::runtime_error(
            "The number of channels in the kernel must match the number of channels in the input matrix"
        );

    const size_t oH = iH - kH + 1;
    const size_t oW = iW - kW + 1;


    Tensor<ValT> output(oH, oW, kCh, iBatchsize);

    for (size_t batch_idx = 0; batch_idx < iBatchsize; ++batch_idx)
    for (size_t channel_idx = 0; channel_idx < kCh; ++channel_idx)
    {
        const auto& iLayer = input (batch_idx, channel_idx);
        const auto& kLayer = kernel(0,         channel_idx);

        for (size_t y = 0; y < oH; ++y)
        for (size_t x = 0; x < oW; ++x)
        {
            ValT acc{};

            for (size_t ky = 0; ky < kH; ++ky)
            for (size_t kx = 0; kx < kW; ++kx)
                acc += iLayer[x + kx, y + ky] * kLayer[kx, ky];

            output(batch_idx, channel_idx)[x, y] += acc;
        }
    }

    return output;
}

namespace json = boost::json;

export template<typename ValT>
decltype(auto) dump(const Tensor<ValT>& tensor)
{
    const size_t channels = tensor.channels();
    const size_t batchSize = tensor.batchSize();

    json::object tensor_obj;
    tensor_obj["height"] = tensor.height();
    tensor_obj["height"] = tensor.height();
    tensor_obj["channels"] = channels;
    tensor_obj["batchSize"] = batchSize;

    json::array layers_arr;
    for (size_t batch_idx = 0; batch_idx < batchSize; ++batch_idx)
        for (size_t channel_idx = 0; channel_idx < channels; ++channel_idx)
            layers_arr.push_back(tensor(batch_idx, channel_idx));

    tensor_obj["layers"] = layers_arr;


    std::cout << "Tensor data (json format): \n"
    std::cout << json::serialize(tensor_obj) << '\n';
}



namespace traits
{

template<typename T>
concept Tensor = requires(T input, T kernel)
{
    { conv_naive(input, kernel) } -> std::same_as<T>;
    // { conv_winograd(input, kernel) } -> std::same_as<T>;
};

}

static_assert(traits::Tensor<Tensor<float>>);

} // namespace tensor