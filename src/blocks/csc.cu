// File: csc.cu
// Description: CSC CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <stdexcept>

#include "blocks/csc.hpp"

namespace
{

constexpr CscKernelParams kCscParams = {
    .matrix = { { 66, 129, 25 },      // Y x 256
                { -38, -74, 112 },    // Cb x 256
                { 112, -94, -18 } },  // Cr x 256
    .bias = { 16, 128, 128 },
};

}  // namespace

CscBlock::CscBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(kCscParams)
{
}

__global__ void
csc_kernel(FrameView<uint8_t> in, FrameView<uint8_t> out, CscKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    const int32_t r = in.at(y, x, 0);
    const int32_t g = in.at(y, x, 1);
    const int32_t b = in.at(y, x, 2);

#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        const int32_t converted =
            ((r * params.matrix[c][0] + g * params.matrix[c][1] + b * params.matrix[c][2]) >> 8) +
            params.bias[c];
        out.at(y, x, c) = static_cast<uint8_t>(max(0, min(converted, 255)));
    }
}

void
CscBlock::execute(PipelineData& data)
{
    if (!data.rgb_sdr)
        throw std::runtime_error("CscBlock: rgb_sdr frame is null");

    auto& frame = *data.rgb_sdr;
    auto out =
        std::make_unique<PitchedFrame<uint8_t>>(frame.width(), frame.height(), frame.channels());

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    csc_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.ycbcr = std::move(out);
}
