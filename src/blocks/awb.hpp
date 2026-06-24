// File: awb.hpp
// Description: Auto white balance (AWB) ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>

#include "isp_block.hpp"

enum AwbGains
{
    RGGB_R = 0,
    RGGB_GR = 1,
    RGGB_GB = 2,
    RGGB_B = 3,

    BGGR_B = 0,
    BGGR_GB = 1,
    BGGR_GR = 2,
    BGGR_R = 3,
};

struct AwbKernelParams
{
    uint32_t rggb_gains[4];
    uint32_t bggr_gains[4];

    bool is_rggb;
    uint32_t sat_value;
};

class AwbBlock : public IspBlock
{
  public:
    explicit AwbBlock(const IspConfig& cfg, cudaStream_t stream = 0);

    void execute(PipelineData& data) override;
    const char*
    name() const override
    {
        return "awb";
    }

  private:
    AwbKernelParams params_;
};
