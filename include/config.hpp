// File: config.hpp
// Description: ISP pipeline configuration types and TOML loader
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <toml++/impl/forward_declarations.hpp>

struct HardwareConfig
{
    uint32_t raw_width;
    uint32_t raw_height;
    uint32_t raw_bit_depth;
    std::string bayer_pattern;
};

struct IspConfig
{
    std::unordered_map<std::string, bool> block_enable_status;
    HardwareConfig hardware;

    IspConfig();
    ~IspConfig();
    IspConfig(IspConfig&&) noexcept;
    IspConfig& operator=(IspConfig&&) noexcept;
    IspConfig(const IspConfig&) = delete;
    IspConfig& operator=(const IspConfig&) = delete;

    // for aaf, csc, this function will return a nullptr because they have
    // no params. Otherwise, it will return the module params as a table.
    // Only meaningful to built-in blocks; dereferencing the result requires
    // <toml++/toml.hpp>, which this header does not pull in.
    const toml::table* block_params(const std::string& name) const;

    static IspConfig load(const std::string& path);

  private:
    std::unique_ptr<toml::table> raw_;
};
