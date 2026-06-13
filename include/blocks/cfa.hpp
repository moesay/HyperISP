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
