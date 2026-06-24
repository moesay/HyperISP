// File: cnf.hpp
// Description: Chroma noise filter (CNF) ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>

#include "isp_block.hpp"

struct CnfKernelParams
{
    int32_t diff_threshold;
    int32_t r_gain;
    int32_t b_gain;

    uint8_t r_row, r_col;
    uint8_t gr_row, gr_col;
    uint8_t gb_row, gb_col;
    uint8_t b_row, b_col;

    uint32_t sat_value;
    uint8_t luma_shift;
};

class CnfBlock : public IspBlock
{
  public:
    explicit CnfBlock(const IspConfig& cfg, cudaStream_t stream = 0);

    void execute(PipelineData& data) override;

    const char*
    name() const override
    {
        return "cnf";
    }

  private:
    CnfKernelParams params_;
};
