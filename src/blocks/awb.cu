#include <cctype>
#include <stdexcept>

#include "blocks/awb.hpp"

namespace
{

AwbKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("awb");
    if (!t)
    {
        throw std::runtime_error("AwbBlock: missing [awb] config sectio");
    }

    auto get_int = [&](const char* key) -> uint32_t
    { return static_cast<uint32_t>((*t)[key].value<uint64_t>().value_or(0)); };

    AwbKernelParams kernel_params;

    kernel_params.rggb_gains[AwbGains::RGGB_R] = get_int("r_gain");
    kernel_params.rggb_gains[AwbGains::RGGB_GR] = get_int("gr_gain");
    kernel_params.rggb_gains[AwbGains::RGGB_GB] = get_int("gb_gain");
    kernel_params.rggb_gains[AwbGains::RGGB_B] = get_int("b_gain");

    kernel_params.bggr_gains[AwbGains::BGGR_B] = get_int("b_gain");
    kernel_params.bggr_gains[AwbGains::BGGR_GB] = get_int("gb_gain");
    kernel_params.bggr_gains[AwbGains::BGGR_GR] = get_int("gr_gain");
    kernel_params.bggr_gains[AwbGains::BGGR_R] = get_int("r_gain");

    /*
       The saturation value calculation should consider the status of the BLC block (enabled or not)
       but for now, KISS and set it to the maximum pixel saturation value
    */
    kernel_params.sat_value = static_cast<uint32_t>((1u << cfg.hardware.raw_bit_depth) - 1);
    kernel_params.is_rggb = std::tolower(cfg.hardware.bayer_pattern[0]) == 'r';

    return kernel_params;
}

}  // namespace

__global__ void
awb_kernel(FrameView<uint16_t> frame, AwbKernelParams params)
{
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= frame.width || y >= frame.height)
        return;
    const uint8_t bayer_channel = (y & 1) * 2 + (x & 1);

    if (params.is_rggb)
    {
        frame.at(y, x) =
            min(params.sat_value, (frame.at(y, x) * params.rggb_gains[bayer_channel] >> 10));
    }
    else
    {
        frame.at(y, x) =
            min(params.sat_value, (frame.at(y, x) * params.bggr_gains[bayer_channel] >> 10));
    }
}

AwbBlock::AwbBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

void
AwbBlock::execute(PipelineData& data)
{
    if (!data.bayer)
        throw std::runtime_error("AwbBlock: bayer frame is null");

    auto& frame = *data.bayer;

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    awb_kernel<<<grid, block, 0, stream_>>>(frame.view(), params_);
}
