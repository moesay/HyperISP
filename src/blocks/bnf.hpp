// File: bnf.hpp
// Description: Bilateral Noise Filter (BNF) ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>

#include "isp_block.hpp"

struct BnfKernelParams
{
    int32_t kernel_size;
    float spatial_sigma;
    float intensity_sigma;

    uint32_t lut_size;
    const int32_t* spatial_weights_lut;
};

class BnfBlock : public IspBlock
{
  public:
    explicit BnfBlock(const IspConfig& cfg, cudaStream_t stream = 0);
    ~BnfBlock() override;

    void execute(PipelineData& data) override;

    const char*
    name() const override
    {
        return "bnf";
    }

  private:
    BnfKernelParams params_;
};
