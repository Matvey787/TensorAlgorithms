module;

#include <iostream>
#include <vector>  
#include <exception> 
#include <memory>
#include <optional>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <mutex>

#ifdef WITH_OPENCL

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 300

#ifndef CL_MAKE_VERSION
#define CL_MAKE_VERSION(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))
#endif

#include "opencl.hpp"

#endif

export module opencl;

#ifdef WITH_OPENCL

namespace opencl
{

export class IDeviceSearcher
{
public:
    virtual cl::Device 
    getDevice(size_t platformIdx, size_t deviceIdx) const = 0;

    virtual cl::Device
    getFirstSuitableDevice() const = 0;
    
    virtual size_t
    getPlatformsCount() const = 0;
    
    virtual size_t
    getDevicesCount(size_t platformIdx) const = 0;
    
    virtual void
    showAllDevicesInfo() const = 0;
    
    virtual ~IDeviceSearcher() = default;
};

export void
printDeviceInfo(const cl::Device& device);

class DeviceSearcher final : public IDeviceSearcher
{
    struct PlatformData
    {
        cl::Platform platform;
        std::vector<cl::Device> devices;
    };

    std::vector<PlatformData> platformsData;

public:
    DeviceSearcher() 
    {
        try
        {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            
            if (platforms.empty())
                throw std::runtime_error("No OpenCL platforms found");
            
            platformsData.reserve(platforms.size());
            
            for (const auto& platform : platforms)
            {
                PlatformData platformData;
                platformData.platform = platform;
                
                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                platformData.devices = std::move(devices);
                
                platformsData.push_back(std::move(platformData));
            }
        }
        catch (const cl::Error& e)
        {
            std::cerr << "OpenCL error during initialization: " 
                      << e.what() << " (" << e.err() << ")" << std::endl;
            throw std::runtime_error("Failed to initialize DeviceSearcher");
        }
    }

    virtual cl::Device
    getDevice(size_t platformIdx, size_t deviceIdx) const override
    {
        if (platformIdx >= platformsData.size())
        {
            throw std::out_of_range(
                "Platform index "                 +
                std::to_string(platformIdx)       + 
                " is out of range. Max index is " + 
                std::to_string(platformsData.size() - 1)
            );
        }

        if (deviceIdx >= platformsData[platformIdx].devices.size())
        {
            throw std::out_of_range(
                "Device index "                  + 
                std::to_string(deviceIdx)        + 
                " is out of range for platform " + 
                std::to_string(platformIdx)      + 
                ". Max index is "                + 
                std::to_string(platformsData[platformIdx].devices.size() - 1)
            );
        }

        return platformsData[platformIdx].devices[deviceIdx];
    }

    virtual cl::Device
    getFirstSuitableDevice() const override
    {
        auto findDevice = [this](cl_device_type preferredType) -> std::optional<cl::Device> 
        {
            cl::Device suitableDevice;

            for (const auto& platformData : platformsData)
            {
                for (const auto& device : platformData.devices)
                {
                    cl_device_type type;
                    device.getInfo(CL_DEVICE_TYPE, &type);

                    std::string deviceName;
                    device.getInfo(CL_DEVICE_NAME, &deviceName);

                    if (type == preferredType)
                        suitableDevice = device;

                    if (deviceName.find("NVIDIA") != std::string::npos)
                    {
                        return device;
                    }
                }
            }

            if (suitableDevice.get())
                return suitableDevice;

            return std::nullopt;
        };
        
        if (auto device_optionalWrap = findDevice(CL_DEVICE_TYPE_GPU))
            return *device_optionalWrap;
        
        if (auto device_optionalWrap = findDevice(CL_DEVICE_TYPE_CPU))
            return *device_optionalWrap;
        
        for (const auto& platformData : platformsData)
        {
            if (!platformData.devices.empty())
                return platformData.devices[0];
        }
        
        throw std::runtime_error("No OpenCL devices available");
    }
    
    virtual size_t
    getPlatformsCount() const override
    {
        return platformsData.size();
    }
    
    virtual size_t
    getDevicesCount(size_t platformIdx) const override
    {
        if (platformIdx >= platformsData.size())
        {
            throw std::out_of_range(
                "Platform index " + std::to_string(platformIdx) + 
                " is out of range"
            );
        }
        
        return platformsData[platformIdx].devices.size();
    }
    
    virtual void
    showAllDevicesInfo() const override
    {
        std::cout << "Found " << platformsData.size() << " OpenCL platform(s)\n\n";
        
        for (size_t i = 0; i < platformsData.size(); ++i)
        {
            std::string platformName;
            platformsData[i].platform.getInfo(CL_PLATFORM_NAME, &platformName);
            
            std::cout << "=== Platform " << i << ": " << platformName 
                      << " ===\n";
            std::cout << "Devices: " << platformsData[i].devices.size() << "\n\n";
            
            for (size_t j = 0; j < platformsData[i].devices.size(); ++j)
            {
                std::cout << "  Device " << j << ":\n";
                printDeviceInfo(platformsData[i].devices[j]);
                std::cout << std::endl;
            }
            
            std::cout << std::string(50, '-') << "\n\n";
        }
    }
};

void
printDeviceInfo(const cl::Device& device)
{
    try
    {
        std::string deviceName;
        std::string deviceVendor;
        cl_device_type deviceType;
        cl_uint maxComputeUnits;
        cl_ulong globalMemSize;
        cl_ulong localMemSize;
        cl_ulong cacheSize;
        size_t maxWorkGroupSize;
        
        device.getInfo(CL_DEVICE_NAME, &deviceName);
        device.getInfo(CL_DEVICE_VENDOR, &deviceVendor);
        device.getInfo(CL_DEVICE_TYPE, &deviceType);
        device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits);
        device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);
        device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
        device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &cacheSize);
        device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
        
        // std::cout << "    Platform index:      " << platformIdx << "\n";
        std::cout << "Name:                " << deviceName << "\n";
        std::cout << "Vendor:              " << deviceVendor << "\n";
        std::cout << "Type:                ";
        
        if (deviceType & CL_DEVICE_TYPE_CPU) std::cout << "CPU ";
        if (deviceType & CL_DEVICE_TYPE_GPU) std::cout << "GPU ";
        if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "Accelerator ";
        if (deviceType & CL_DEVICE_TYPE_DEFAULT) std::cout << "Default ";
        
        std::cout << "\n";
        std::cout << "Max compute units:   " << maxComputeUnits << "\n";
        std::cout << "Global memory:       " 
                    << (globalMemSize / (1024 * 1024)) << " MB\n";
        std::cout << "Local memory:        " 
                    << (localMemSize / 1024) << " KB\n";
        std::cout << "Cache size:          " 
                    << (cacheSize / 1024) << " KB\n";
        std::cout << "Max work-group size: " << maxWorkGroupSize << "\n";
    }
    catch (const cl::Error& e)
    {
        std::cerr << "    Error getting device info: " 
                    << e.what() << " (" << e.err() << ")\n";
    }
}

export std::unique_ptr<IDeviceSearcher>
createDeviceSearcher()
{
    return std::make_unique<DeviceSearcher>();
}


























std::string
readKernel(std::string_view fileWithKernel)
{
    std::ifstream kernelFile(fileWithKernel.data());

    if (!kernelFile.is_open())
    {
        throw std::runtime_error("Can't open kernel file");
    }

    std::stringstream buffer;
    buffer << kernelFile.rdbuf();

    return buffer.str();
}

export class KernelExecutor final
{
    cl::Device       device_;
    cl::Context      context_;
    cl::Program      program_;
    cl::CommandQueue queue_;

    mutable std::mutex queueGuard_;

    class BufferData
    {
    public:
        cl::Buffer          clBuff;
        cl_mem_flags        flags;
        std::size_t         size;
    };

    std::unordered_map<std::string, cl::Kernel> kernelsData_;
    std::unordered_map<std::string, BufferData> buffersData_;

public:

    KernelExecutor(std::string_view filePath) :
        device_ (opencl::createDeviceSearcher()->getFirstSuitableDevice()),
        context_(device_),
        program_(context_, readKernel(filePath)),
        queue_  (context_, device_)
    {
        std::cout << "Selected device for convolution:\n";
        opencl::printDeviceInfo(device_);

        try
        {
            program_.build();
        }
        catch (...)
        {
            std::string buildLog = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);
            std::cerr << "Build failed:\n" << buildLog << std::endl;
            throw;
        }
    }

    cl::Kernel& registerKernel(std::string_view kernelName)
    {
        std::string name(kernelName);

        auto [it, inserted] = kernelsData_.emplace(name, cl::Kernel{});

        if (inserted)
            it->second = cl::Kernel(program_, name.c_str());

        return it->second;
    }

    cl::Kernel& getKernel(std::string_view kernelName)
    {
        std::string name(kernelName);

        auto it = kernelsData_.find(name);

        if (it == kernelsData_.end())
            throw std::runtime_error("Kernel '" + name + "' is not registered.");

        return it->second;
    }

    void registerBuffer(std::string_view  buffName,
                        cl_mem_flags      flags,
                        std::size_t       buffSize)
    {
        if (flags & CL_MEM_COPY_HOST_PTR)
            throw std::runtime_error("An attempt to register a buffer with the CL_MEM_COPY_HOST_PTR flag without a pointer to the host buffer.");

        std::string name(buffName);

        buffersData_.emplace(std::string(buffName), BufferData{
            .clBuff   = cl::Buffer(context_, flags, sizeof(float) * buffSize),
            .flags    = flags,
            .size     = buffSize
        });
    }

    void registerBuffer(std::string_view       buffName,
                        cl_mem_flags           flags,
                        const std::vector<float>&    hostData)
    {
        std::string name(buffName);

        BufferData buffData;
        buffData.flags    = flags;
        buffData.size     = hostData.size();
        buffData.clBuff   = cl::Buffer(
            context_,
            flags,
            sizeof(float) * buffData.size,
            const_cast<float*>(hostData.data())
        );

        buffersData_.emplace(name, std::move(buffData));
    }

    void updateBuffer(std::string_view buffName,
                  cl_mem_flags     flags,
                  const std::vector<float>& hostData)
    {
        std::string name(buffName);

        BufferData buffData;
        buffData.flags  = flags;
        buffData.size   = hostData.size();
        buffData.clBuff = cl::Buffer(
            context_,
            flags,
            sizeof(float) * buffData.size,
            const_cast<float*>(hostData.data())
        );

        buffersData_.insert_or_assign(name, std::move(buffData));
    }

    void updateBuffer(std::string_view buffName,
                    cl_mem_flags     flags,
                    std::size_t      buffSize)
    {
        if (flags & CL_MEM_COPY_HOST_PTR)
            throw std::runtime_error("...");

        std::string name(buffName);

        buffersData_.insert_or_assign(name, BufferData{
            .clBuff = cl::Buffer(context_, flags, sizeof(float) * buffSize),
            .flags  = flags,
            .size   = buffSize
        });
    }

    cl::Buffer& getClBuffer(std::string_view buffName)
    {
        auto it = buffersData_.find(std::string(buffName));

        if (it == buffersData_.end())
            throw std::runtime_error("Buffer " + std::string(buffName) + "   is not registered.");

        return it->second.clBuff;
    }

    void fetchBuffer(std::string_view buffName, std::vector<float>& dst)
    {
        auto it = buffersData_.find(std::string(buffName));

        if (it == buffersData_.end())
            throw std::runtime_error("Buffer " + std::string(buffName) + " is not registered.");

        BufferData& buffData = it->second;

        dst.resize(buffData.size);

        {
            std::lock_guard lock(queueGuard_);
            queue_.enqueueReadBuffer(
                buffData.clBuff,
                CL_TRUE,
                0,
                sizeof(float) * buffData.size,
                dst.data()
            );
        }
    }

    void enqueueNDRange(cl::Kernel& kernel, cl::NDRange global) const
    {
        std::lock_guard lock(queueGuard_);
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    }

    void finish()
    {
        std::lock_guard lock(queueGuard_);
        queue_.finish();
    }

    const cl::Context& context() const { return context_; }
    const cl::Device&  device()  const { return device_;  }
    cl::CommandQueue&  queue() { return queue_;  }
};


}; // namespace opencl

#endif








