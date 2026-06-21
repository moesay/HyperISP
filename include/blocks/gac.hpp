// File: gac.hpp
// Description: Gamma correction (GAC) ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>

#include "isp_block.hpp"

struct GacKernelParams
{
    uint32_t gain;
    uint32_t hdr_sat;

    const uint8_t* lut;
};

class GacBlock : public IspBlock
{
  public:
    explicit GacBlock(const IspConfig& cfg, cudaStream_t stream = 0);
    ~GacBlock() override;

    void execute(PipelineData& data) override;

    const char*
    name() const override
    {
        return "gac";
    }

  private:
    GacKernelParams params_;
};
