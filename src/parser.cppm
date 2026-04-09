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
            ("h,help", "Show help message");

        result_ = options_.parse(argc, argv);
    }

    bool hasOption(std::string_view optionName) const
    {
        return result_.count(std::string(optionName)) > 0;
    }

    std::string getOptionVal(std::string_view optionName) const
    {
        if (hasOption(optionName))
        {
            return result_[std::string(optionName)].as<std::string>();
        }

        return "";
    }

    const cxxopts::Options& getOptions()
    {
        return options_;
    }

};

} // namespace parser
