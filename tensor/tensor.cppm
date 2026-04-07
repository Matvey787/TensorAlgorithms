module;

#define BOOST_JSON_HEADER_ONLY
#include <boost/json.hpp>

#include <exception>
#include <concepts>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

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
    const size_t iH        = input.height();
    const size_t iW        = input.width();
    const size_t iBatch    = input.batchSize();
    const size_t kH        = kernel.height();
    const size_t kW        = kernel.width();
    const size_t iCh       = input.channels();
    const size_t kCh       = kernel.channels();

    if (kH > iH || kW > iW) [[unlikely]]
        throw std::runtime_error("Kernel larger than input.");

    if (iCh != kCh) [[unlikely]]
        throw std::runtime_error(
            "The number of channels in the kernel must match the number of channels in the input."
        );

    const size_t oH = iH - kH + 1;
    const size_t oW = iW - kW + 1;

    Tensor<ValT> output(oH, oW, kCh, iBatch);

    for (size_t batch_idx = 0; batch_idx < iBatch; ++batch_idx)
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

template<typename ValT>
Tensor<ValT> json_to_tensor(const json::object& obj);

export template<typename ValT>
void dump(const std::vector<Tensor<ValT>>& tensors, std::ostream& os = std::cout)
{
    json::array tensors_arr;
    for (const auto& tensor : tensors)
    {
        const size_t channels  = tensor.channels();
        const size_t batchSize = tensor.batchSize();
        const size_t h         = tensor.height();
        const size_t w         = tensor.width();

        json::object tensor_obj;
        tensor_obj["height"]    = h;
        tensor_obj["width"]     = w;
        tensor_obj["channels"]  = channels;
        tensor_obj["batchSize"] = batchSize;

        json::array layers_arr;
        for (size_t batch_idx = 0; batch_idx < batchSize; ++batch_idx)
        for (size_t channel_idx = 0; channel_idx < channels; ++channel_idx)
        {
            const auto& layer = tensor(batch_idx, channel_idx);
            json::array flat;
            for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                flat.push_back(layer[x, y]);
            layers_arr.push_back(flat);
        }

        tensor_obj["layers"] = layers_arr;
        tensors_arr.push_back(tensor_obj);
    }

    json::object root;
    root["tensors"] = tensors_arr;
    os << json::serialize(root) << '\n';
}

export template<typename ValT>
void dump(const Tensor<ValT>& tensor, std::ostream& os = std::cout)
{
    dump(std::vector<Tensor<ValT>>{tensor}, os);
}

export template<typename ValT>
std::vector<Tensor<ValT>> read(const std::string& jsonFileName)
{
    std::ifstream ifs(jsonFileName);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open file: " + jsonFileName);

    std::stringstream buff;
    buff << ifs.rdbuf();   // buff << ifs не читает содержимое, нужно rdbuf()

    const std::string str = buff.str();
    const auto val = json::parse(str);
    const auto& arr = val.as_object().at("tensors").as_array();

    std::vector<Tensor<ValT>> result;
    result.reserve(arr.size());

    for (const auto& item : arr)
        result.push_back(json_to_tensor<ValT>(item.as_object()));

    return result;
}

template<typename ValT>
Tensor<ValT> json_to_tensor(const json::object& obj)
{
    const size_t h         = obj.at("height").as_int64();
    const size_t w         = obj.at("width").as_int64();
    const size_t channels  = obj.at("channels").as_int64();
    const size_t batchSize = obj.at("batchSize").as_int64();

    const size_t layerSize  = h * w;
    const size_t layerCount = channels * batchSize;

    Tensor<ValT> tensor(h, w, channels, batchSize);

    const auto& layers = obj.at("layers").as_array();

    if (layers.size() != layerCount)
        throw std::runtime_error("Layer count must equal channels * batchSize.");

    for (size_t layer_idx = 0; layer_idx < layerCount; ++layer_idx)
    {
        const auto& json_layer = layers[layer_idx].as_array();

        if (json_layer.size() != layerSize)
            throw std::runtime_error("Layer element count must equal height * width.");

        auto& tensor_layer = tensor(layer_idx / channels, layer_idx % channels);

        for (size_t elem_idx = 0; elem_idx < layerSize; ++elem_idx)
            tensor_layer[elem_idx % w, elem_idx / w] =
                static_cast<ValT>(json_layer[elem_idx].as_double());
    }

    return tensor;
}

namespace traits
{

template<typename T>
concept TensorConcept = requires(T input, T kernel)
{
    { conv_naive(input, kernel) } -> std::same_as<T>;
};

}

static_assert(traits::TensorConcept<Tensor<float>>);

} // namespace tensor
