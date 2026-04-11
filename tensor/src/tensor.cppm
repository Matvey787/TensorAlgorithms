module;

#include <concepts>
#include <vector>
#include <stdexcept>
#include <numeric>

export module tensor_gen;

namespace tensor
{

export template<typename ValT = float>
class Layer final
{
    size_t height_;
    size_t width_;
    std::vector<ValT> data_;

public:
    Layer(size_t h, size_t w)
        : height_(h), width_(w), data_(h * w, ValT{})
    {}

    Layer(std::initializer_list<std::initializer_list<ValT>> rawLayer)
    {
        if (rawLayer.size() == 0) return;

        height_ = rawLayer.size();
        width_  = rawLayer.begin()->size();
        data_.resize(height_ * width_);

        size_t row_idx = 0;
        for (const auto& row : rawLayer)
        {
            if (row.size() != width_) [[unlikely]]
                throw std::runtime_error("Initializer list rows have inconsistent widths.");
            
            size_t col_idx = 0;
            for (const auto& val : row)
            {
                (*this)[col_idx, row_idx] = val;
                ++col_idx;
            }
            ++row_idx;
        }
    }

    Layer(
        std::vector<ValT>::iterator start,
        std::vector<ValT>::iterator end,
        size_t                      height
    ) : height_{height},
        width_ {static_cast<size_t>(std::distance(start, end) / height)},
        data_  {start, end}
    {
    }

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

    Layer mul_matrix(const Layer& other) const
    {
        if (width_ != other.height_) [[unlikely]]
            throw std::runtime_error("Matrix multiplication dimension mismatch: left width must equal right height");
        
        Layer result(height_, other.width_);
        
        for (size_t y = 0; y < height_; ++y)
        {
            for (size_t x = 0; x < other.width_; ++x)
            {
                ValT sum = ValT{};
                for (size_t k = 0; k < width_; ++k)
                {
                    sum += (*this)[k, y] * other[x, k];
                }
                result[x, y] = sum;
            }
        }
        
        return result;
    }

    Layer mul_elementWise(const Layer& other) const
    {

        if (width_ != other.width_ || height_ != other.height_)
            throw std::runtime_error("Element-wise multiplication dimension mismatch: dimensions of layers should be equal");
            
        Layer result(width_, height_);
        
        for (size_t y = 0; y < height_; ++y)
            for (size_t x = 0; x < width_; ++x)
                result[x, y] = (*this)[x, y] * other[x, y];

        return result;
    }

    Layer add(const Layer& other) const
    {
        if (width_ != other.width_ || height_ != other.height_)
            throw std::runtime_error("Layer addition dimension mismatch: dimensions of layers should be equal");

        Layer result(width_, height_);

        for (size_t y = 0; y < height_; ++y)
            for (size_t x = 0; x < width_; ++x)
                result[x, y] = (*this)[x, y] + other[x, y];
        
        return result;
    }

    Layer transpose() const
    {
        Layer result(width_, height_);

        for (size_t y = 0; y < height_; ++y)
            for (size_t x = 0; x < width_; ++x)
                result[y, x] = (*this)[x, y];
 
        return result;
    }

    ValT sum() const
    {
        return std::accumulate(data_.begin(), data_.end(), ValT{});
    }

    const std::vector<ValT>& data() const noexcept
    {
        return data_;
    }

    [[nodiscard]] size_t height() const noexcept { return height_; }
    [[nodiscard]] size_t width()  const noexcept { return width_;  }
};



export template<typename ValT = float>
class Tensor final
{
    size_t height_{0};
    size_t width_{0};
    size_t channels_{0};
    size_t batchSize_{0};
    std::vector<Layer<ValT>> layers_;

public:
    Tensor() = default;

    Tensor(size_t h, size_t w, size_t c, size_t batch)
        : height_(h), width_(w), channels_(c), batchSize_(batch)
    {
        layers_.reserve(c * batch);
        for (size_t i = 0; i < c * batch; ++i)
            layers_.emplace_back(h, w);
    }

    Tensor(
        std::vector<ValT>&& flatData,
        std::size_t       height,
        std::size_t       width,
        std::size_t       channels,
        std::size_t       batchSize
    ) : height_   (height),
        width_    (width),
        channels_ (channels),
        batchSize_(batchSize)
    {
        if (flatData.size() != height * width * channels * batchSize)
            throw std::runtime_error("Flat data size does not match tensor dimensions.");

        layers_.reserve(channels * batchSize);
        auto it = flatData.begin();

        for (std::size_t i = 0; i < channels * batchSize; ++i)
        {
            auto next_it = it + height * width;
            layers_.emplace_back(it, next_it, height);
            it = next_it;
        }
    }

    [[nodiscard]] size_t height()    const noexcept { return height_;    }
    [[nodiscard]] size_t width()     const noexcept { return width_;     }
    [[nodiscard]] size_t channels()  const noexcept { return channels_;  }
    [[nodiscard]] size_t batchSize() const noexcept { return batchSize_; }

    void addLayer(Layer<ValT>&& newLayer)
    {
        if (layers_.empty())
        {
            height_    = newLayer.height();
            width_     = newLayer.width();
            channels_  = 1;
            batchSize_ = 1;
        }
        else
        {
            if (newLayer.height() != height_ || newLayer.width() != width_)
                throw std::runtime_error("Layer dimensions must match tensor dimensions");
        }
        
        layers_.push_back(std::move(newLayer));

        batchSize_ = layers_.size() / channels_;
    }

    Layer<ValT>& operator()(size_t batch_idx, size_t channel_idx)
    {
        return layers_[batch_idx * channels_ + channel_idx];
    }

    const Layer<ValT>& operator()(size_t batch_idx, size_t channel_idx) const
    {
        return layers_[batch_idx * channels_ + channel_idx];
    }


    
    // for opencl global buffers

    std::vector<ValT> data() const
    {
        std::vector<ValT> rawData;
        rawData.reserve(batchSize_ * channels_ * width_ * height_);

        for (std::size_t i = 0; i < channels_ * batchSize_; ++i)
        {
            auto&& layerData = layers_[i].data();
            std::copy(layerData.begin(), layerData.end(), std::back_inserter(rawData));
        }

        return rawData;
    }

    size_t size() const noexcept
    {
        return batchSize_ * channels_ * width_ * height_;
    }
};

} // namespace tensor
