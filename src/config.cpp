// File: config.cpp
// Description: IspConfig TOML loading implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <stdexcept>
#include <string>

#include <toml++/toml.hpp>

#include <config.hpp>

IspConfig::IspConfig() = default;
IspConfig::~IspConfig() = default;
IspConfig::IspConfig(IspConfig&&) noexcept = default;
IspConfig& IspConfig::operator=(IspConfig&&) noexcept = default;

const toml::table*
IspConfig::block_params(const std::string& name) const
{
    if (!raw_)
    {
        return nullptr;
    }

    if (auto* node = raw_->get(name))
    {
        return node->as_table();
    }
    return nullptr;
}

IspConfig
IspConfig::load(const std::string& path)
{
    toml::table tbl;

    try
    {
        tbl = toml::parse_file(path);
    }
    catch (const toml::parse_error& e)
    {
        throw std::runtime_error(std::string("Failed to parse config: ") + std::string(e.what()));
    }

    IspConfig cfg;

    if (auto* status = tbl["block_enable_status"].as_table())
    {
        for (auto& [key, val] : *status)
        {
            if (auto* b = val.as_boolean())
                cfg.block_enable_status[std::string(key)] = b->get();
        }
    }
    else
    {
        throw std::runtime_error("Config missing [block_enable_status]");
    }

    auto hw = tbl["hardware"];
    auto require_uint = [&](const toml::node_view<toml::node> node, const char* key) -> uint32_t
    {
        if (auto v = node[key].value<int64_t>())
        {
            return static_cast<uint32_t>(*v);
        }
        throw std::runtime_error(std::string("Config missing hardware.") + key);
    };

    cfg.hardware.raw_width = require_uint(hw, "raw_width");
    cfg.hardware.raw_height = require_uint(hw, "raw_height");
    cfg.hardware.raw_bit_depth = require_uint(hw, "raw_bit_depth");

    if (auto v = hw["bayer_pattern"].value<std::string>())
    {
        cfg.hardware.bayer_pattern = *v;
    }
    else
    {
        throw std::runtime_error("Config missing hardware.bayer_pattern");
    }

    cfg.raw_ = std::make_unique<toml::table>(std::move(tbl));

    return cfg;
}
