// File: csc.hpp
// Description: Color space conversion (CSC) ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>

#include "isp_block.hpp"

struct CscKernelParams
{
    int32_t matrix[3][3];
    int32_t bias[3];
};

class CscBlock : public IspBlock
{
  public:
    explicit CscBlock(const IspConfig& cfg, cudaStream_t stream = 0);

    void execute(PipelineData& data) override;

    const char*
    name() const override
    {
        return "csc";
    }

  private:
    CscKernelParams params_;
};
