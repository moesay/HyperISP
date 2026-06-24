// File: ccm.hpp
// Description: Color correction matrix (CCM) ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>

#include "isp_block.hpp"

struct CcmKernelParams
{
    int32_t matrix[3][3];
    int32_t bias[3];

    uint32_t sat_value;
};

class CcmBlock : public IspBlock
{
  public:
    explicit CcmBlock(const IspConfig& cfg, cudaStream_t stream = 0);

    void execute(PipelineData& data) override;

    const char*
    name() const override
    {
        return "ccm";
    }

  private:
    CcmKernelParams params_;
};
