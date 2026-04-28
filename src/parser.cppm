module;

#include <cxxopts.hpp>
#include <iostream>
#include <fstream>
#include <string>

export module parser;

namespace parser
{

export class Parser
{
    cxxopts::Options options_;
    cxxopts::ParseResult result_;

public:

    Parser(int argc, char** argv) : options_("winograd", "Tensor convolution program")
    {
        options_.add_options()
            ("s,source", "Input source file", cxxopts::value<std::string>())
            ("o,output", "Output JSON file name", cxxopts::value<std::string>())
            ("h,help", "Show help message")
            ("gpu", "Use GPU", cxxopts::value<bool>()->default_value("false"))
            ("winograd", 
                "Use Winograd convolution algoritm. (Supported kernel is 3 * 3)", 
                cxxopts::value<bool>()->default_value("false"))
            ("im2col", 
                "Use Im2Col convolution algoritm.", 
                cxxopts::value<bool>()->default_value("false"))
            ("naive", 
                "Use naive convolution algorithm.", 
                cxxopts::value<bool>()->default_value("false"));

        result_ = options_.parse(argc, argv);
    }

    bool hasOption(std::string_view optionName) const
    {
        return result_.count(std::string(optionName)) > 0;
    }

    std::optional<cxxopts::OptionValue> getOptionVal(std::string_view optionName) const
    {
        if (hasOption(optionName)) return result_[std::string(optionName)];

        return std::nullopt;
    }

    const cxxopts::Options& getOptions()
    {
        return options_;
    }

};

} // namespace parser
