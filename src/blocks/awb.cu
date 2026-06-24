// File: awb.cu
// Description: AWB CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <stdexcept>

#include <toml++/toml.hpp>

#include "awb.hpp"

namespace
{

AwbKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("awb");
    if (!t)
    {
        throw std::runtime_error("AwbBlock: missing [awb] config section");
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

/*
   The logic used to get the bayer channel without branching is based on the fact that
   CUDA threads are contiguous, so we can determine the channel by checking the parity of both
   indices Each term of these (x&1) and (y&1) will give either 0 or 1; the trick is combining them
   together.

   x=0   x=1   x=2   x=3   x=4    ...
   y=0    0     1     0     1     0     <- R  Gr  R  Gr ...
   y=1    2     3     2     3     2     <- Gb B   Gb B  ...
   y=2    0     1     0     1     0
   y=3    2     3     2     3     2

   Ponder the grid above and you will see a pattern. I don't wanna go so verbose but consider the 4
   possibilities (both even, both odd, x odd; y even, x even; y odd) and you will see the light

   So the way is turning the y (whatever it is) to be the high bit (shift or mull) and glue the x
   bit as a low bit

   This will give us c = (y & 1) << 1 | (x & 1)
   Or, c = (y & 1) * 2 + (x & 1)
*/
__global__ void
awb_kernel(FrameView<uint16_t> frame, AwbKernelParams params)
{
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= frame.width || y >= frame.height)
        return;
    const uint8_t bayer_channel = (y & 1) << 1 | (x & 1);

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
