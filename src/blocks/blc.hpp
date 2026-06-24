// File: blc.hpp
// Description: Black level correction (BLC) ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>

#include "isp_block.hpp"

struct BlcKernelParams
{
    int32_t bl_r, bl_gr, bl_gb, bl_b;
    int32_t alpha, beta;
    uint32_t sat_value;
    // Position of each pixel in the 2x2 bayer tile
    uint8_t r_row, r_col;
    uint8_t gr_row, gr_col;
    uint8_t gb_row, gb_col;
    uint8_t b_row, b_col;
};

class BlcBlock : public IspBlock
{
  public:
    explicit BlcBlock(const IspConfig& cfg, cudaStream_t stream = 0);

    void execute(PipelineData& data) override;
    const char*
    name() const override
    {
        return "blc";
    }

  private:
    BlcKernelParams params_;
};
