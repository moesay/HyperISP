// File: nlm.hpp
// Description: Non-local means (NLM) luma denoising ISP block declaration
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <cstdint>

#include "isp_block.hpp"

struct NlmKernelParams
{
    int32_t search_window_size;
    int32_t patch_size;
    int32_t lut_size;

    const int32_t* weights_lut;
};

class NlmBlock : public IspBlock
{
  public:
    explicit NlmBlock(const IspConfig& cfg, cudaStream_t stream = 0);
    ~NlmBlock() override;

    void execute(PipelineData& data) override;

    const char*
    name() const override
    {
        return "nlm";
    }

  private:
    NlmKernelParams params_;
};
