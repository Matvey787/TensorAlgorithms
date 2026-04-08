module;

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>

export module tensor_io;

import tensor_gen;

namespace tensor
{

using json = nlohmann::json;

template<typename ValT>
Tensor<ValT> json_to_tensor(const json& obj)
{
    const size_t h         = obj.at("height");
    const size_t w         = obj.at("width");
    const size_t channels  = obj.at("channels");
    const size_t batchSize = obj.at("batchSize");

    const size_t layerSize  = h * w;
    const size_t layerCount = channels * batchSize;

    Tensor<ValT> tensor(h, w, channels, batchSize);

    const auto& layers = obj.at("layers");

    if (layers.size() != layerCount)
        throw std::runtime_error("Layer count must equal channels * batchSize.");

    for (size_t layer_idx = 0; layer_idx < layerCount; ++layer_idx)
    {
        const auto& json_layer = layers[layer_idx];

        if (json_layer.size() != layerSize)
            throw std::runtime_error("Layer element count must equal height * width.");

        auto& tensor_layer = tensor(layer_idx / channels, layer_idx % channels);

        for (size_t elem_idx = 0; elem_idx < layerSize; ++elem_idx)
            tensor_layer[elem_idx % w, elem_idx / w] =
                static_cast<ValT>(json_layer[elem_idx].template get<double>());
    }

    return tensor;
}

export template<typename ValT>
std::vector<Tensor<ValT>> read(const std::string& jsonFileName)
{
    std::ifstream ifs(jsonFileName);
    
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open file: " + jsonFileName);

    const json root = json::parse(ifs);

    std::vector<Tensor<ValT>> result;
    const auto& arr = root.at("tensors");
    result.reserve(arr.size());

    for (const auto& item : arr)
        result.push_back(json_to_tensor<ValT>(item));

    return result;
}

export template<typename ValT>
void dump(const std::vector<Tensor<ValT>>& tensors, std::ostream& os = std::cout)
{
    nlohmann::ordered_json root;
    root["tensors"] = json::array();

    for (const auto& tensor : tensors)
    {
        const size_t h         = tensor.height();
        const size_t w         = tensor.width();
        const size_t channels  = tensor.channels();
        const size_t batchSize = tensor.batchSize();

        nlohmann::ordered_json tensor_obj;
        tensor_obj["height"]    = h;
        tensor_obj["width"]     = w;
        tensor_obj["channels"]  = channels;
        tensor_obj["batchSize"] = batchSize;
        tensor_obj["layers"]    = json::array();

        for (size_t batch_idx = 0; batch_idx < batchSize; ++batch_idx)
        for (size_t channel_idx = 0; channel_idx < channels; ++channel_idx)
        {
            const auto& Layer = tensor(batch_idx, channel_idx);
            nlohmann::ordered_json flat = json::array();

            for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                flat.push_back(Layer[x, y]);

            tensor_obj["layers"].push_back(flat);
        }
 
        root["tensors"].push_back(tensor_obj);
    }

    os << root.dump(4) << '\n';
}

export template<typename ValT>
void dump(const Tensor<ValT>& tensor, std::ostream& os = std::cout)
{
    dump(std::vector<Tensor<ValT>>{tensor}, os);
}

} // namespace tensor

