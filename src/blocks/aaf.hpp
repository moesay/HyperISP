// File: aaf.hpp
// Description: Anti-aliasing filter (AAF) ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once
#include "isp_block.hpp"

struct AafKernelParams
{
};

class AafBlock : public IspBlock
{
  public:
    explicit AafBlock(const IspConfig& cfg, cudaStream_t stream = 0);

    void execute(PipelineData& data) override;
    const char*
    name() const override
    {
        return "aaf";
    }

  private:
    AafKernelParams params_;
};
