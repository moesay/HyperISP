// File: cfa.hpp
// Description: Color filter array (CFA) demosaic block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include "isp_block.hpp"

enum CfaMode
{
    Bilinear,
    Malvar
};

struct CfaKernelParams
{
    bool is_rggb;
    CfaMode mode;
};

class CfaBlock : public IspBlock
{
  public:
    explicit CfaBlock(const IspConfig& cfg, cudaStream_t stream = 0);

    void execute(PipelineData& data) override;

    const char*
    name() const override
    {
        return "cfa";
    }

  private:
    CfaKernelParams params_;
};
