// File: aaf.cu
// Description: AAF CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <memory>
#include <stdexcept>

#include "aaf.hpp"

AafBlock::AafBlock(const IspConfig& cfg, cudaStream_t stream) : IspBlock(cfg, stream) {}

__global__ void
aaf_kernel(FrameView<uint16_t> in, FrameView<uint16_t> out, AafKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    const int32_t curr = in.at(y, x);
    const auto n = in.neighbors8(y, x);

    out.at(y, x) = ((curr * 8) + n.top + n.bottom + n.left + n.right + n.top_left + n.top_right +
                    n.bottom_left + n.bottom_right) >>
                   4;
}

void
AafBlock::execute(PipelineData& data)
{
    if (!data.bayer)
    {
        throw std::runtime_error("AafBlock: bayer frame is null");
    }

    auto& frame = *data.bayer;
    auto out =
        std::make_unique<PitchedFrame<uint16_t>>(frame.width(), frame.height(), frame.channels());

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.y,
                    (frame.height() + block.y - 1) / block.y);

    aaf_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.bayer = std::move(out);
}
