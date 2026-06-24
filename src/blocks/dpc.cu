// File: dpc.cu
// Description: DPC CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <stdexcept>

#include <toml++/toml.hpp>

#include "dpc.hpp"

namespace
{

DpcKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("dpc");
    if (!t)
    {
        throw std::runtime_error("DpcBlock: missing [dpc] config section");
    }

    DpcKernelParams kernel_params{};
    kernel_params.diff_threshold = (*t)["diff_threshold"].value<int64_t>().value_or(0);
    return kernel_params;
}

}  // namespace

/*
   TODO:
   writes go to a separate buffer because a thread's neighbor might be a dead pixels
   being corrected by other thread. Lockstep execution should take care of this
   but it's a deprecated feature so for now, I will keep it like this
*/
__global__ void
dpc_kernel(FrameView<uint16_t> in, FrameView<uint16_t> out, DpcKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    const int32_t curr = in.at(y, x);
    const auto n = in.neighbors8(y, x);

    const int32_t threshold = params.diff_threshold;

    /*
       TODO:
       idk if its good in the IQ field to get the died pixels count or its useless info since
       there is nothing to do but changing the sensor. But maybe in the future i will implement
       this.
    */
    const bool is_dead =
        (abs(curr - n.top) > threshold) && (abs(curr - n.bottom) > threshold) &&
        (abs(curr - n.left) > threshold) && (abs(curr - n.right) > threshold) &&
        (abs(curr - n.top_left) > threshold) && (abs(curr - n.top_right) > threshold) &&
        (abs(curr - n.bottom_left) > threshold) && (abs(curr - n.bottom_right) > threshold);

    uint16_t result = curr;
    if (is_dead)
    {
        const int32_t vert_diff = abs(2 * curr - n.top - n.bottom);
        const int32_t hori_diff = abs(2 * curr - n.left - n.right);
        const int32_t left_diag_diff = abs(2 * curr - n.top_left - n.bottom_right);
        const int32_t right_diag_diff = abs(2 * curr - n.bottom_left - n.top_right);

        int32_t best = vert_diff;
        int32_t avg = (n.top + n.bottom) >> 1;

        if (hori_diff < best)
        {
            best = hori_diff;
            avg = (n.left + n.right) >> 1;
        }
        if (left_diag_diff < best)
        {
            best = left_diag_diff;
            avg = (n.top_left + n.bottom_right) >> 1;
        }
        if (right_diag_diff < best)
        {
            best = right_diag_diff;
            avg = (n.bottom_left + n.top_right) >> 1;
        }

        result = static_cast<uint16_t>(avg);
    }

    out.at(y, x) = result;
}

DpcBlock::DpcBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

void
DpcBlock::execute(PipelineData& data)
{
    if (!data.bayer)
    {
        throw std::runtime_error("DpcBlock: bayer frame is null");
    }

    auto& frame = *data.bayer;
    auto out =
        std::make_unique<PitchedFrame<uint16_t>>(frame.width(), frame.height(), frame.channels());

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    dpc_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.bayer = std::move(out);
}
