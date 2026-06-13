#include <stdexcept>

#include "blocks/cfa.hpp"

namespace
{

CfaKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("cfa");

    if (!t)
    {
        throw std::runtime_error("CfaBlock: missing [cfa] config section");
    }

    CfaKernelParams kernel_params;

    kernel_params.is_rggb = std::tolower(cfg.hardware.bayer_pattern[0]) == 'r';
    kernel_params.mode = ((*t)["mode"].value<std::string>().value_or("")) == "malvar"
                             ? CfaMode::Malvar
                             : CfaMode::Bilinear;

    return kernel_params;
}

}  // namespace

__global__ void
cfa_bilinear_kernel(FrameView<uint16_t> in, FrameView<uint16_t> out, CfaKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    const int32_t curr = in.at(y, x);
    const auto n = in.neighbors8(y, x, 1);
    const uint8_t bayer_channel = (y & 1) << 1 | (x & 1);

    int32_t r, g, b;

    if (bayer_channel == 0 || bayer_channel == 3)
    {
        g = (n.top + n.bottom + n.left + n.right) >> 2;
        const int32_t diag = (n.top_left + n.top_right + n.bottom_left + n.bottom_right) >> 2;

        const bool is_red_site = params.is_rggb ? (bayer_channel == 0) : (bayer_channel == 3);
        if (is_red_site)
        {
            r = curr;
            b = diag;
        }
        else
        {
            b = curr;
            r = diag;
        }
    }
    else
    {
        g = curr;

        const int32_t horizontal = (n.left + n.right) >> 1;
        const int32_t vertical = (n.top + n.bottom) >> 1;

        const bool horizontal_is_red = params.is_rggb ? (bayer_channel == 1) : (bayer_channel == 2);
        if (horizontal_is_red)
        {
            r = horizontal;
            b = vertical;
        }
        else
        {
            b = horizontal;
            r = vertical;
        }
    }

    out.at(y, x, 0) = static_cast<uint16_t>(r);
    out.at(y, x, 1) = static_cast<uint16_t>(g);
    out.at(y, x, 2) = static_cast<uint16_t>(b);
}

CfaBlock::CfaBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

void
CfaBlock::execute(PipelineData& data)
{
    if (!data.bayer)
        throw std::runtime_error("CfaBlock: bayer frame is null");

    if (params_.mode != CfaMode::Bilinear)
        throw std::runtime_error("CfaBlock: only bilinear mode is implemented");

    auto& frame = *data.bayer;
    auto out = std::make_unique<PitchedFrame<uint16_t>>(frame.width(), frame.height(), 3);

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    cfa_bilinear_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.rgb_hdr = std::move(out);
}
