// File: ccm.cu
// Description: CCM CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <stdexcept>

#include "blocks/ccm.hpp"

namespace
{

CcmKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("ccm");
    if (!t)
        throw std::runtime_error("CcmBlock: missing [ccm] config section");

    const toml::array* matrix = (*t)["matrix"].as_array();
    if (!matrix || matrix->size() != 3)
        throw std::runtime_error("CcmBlock: 'matrix' must have 3 rows");

    CcmKernelParams kernel_params;

    for (size_t row = 0; row < 3; ++row)
    {
        const toml::array* row_arr = (*matrix)[row].as_array();
        if (!row_arr || row_arr->size() != 4)
            throw std::runtime_error("CcmBlock: each 'matrix' row must have 4 columns");

        for (size_t col = 0; col < 3; ++col)
            kernel_params.matrix[row][col] =
                static_cast<int32_t>((*row_arr)[col].value<int64_t>().value_or(0));

        kernel_params.bias[row] = static_cast<int32_t>((*row_arr)[3].value<int64_t>().value_or(0));
    }

    kernel_params.sat_value = (1u << cfg.hardware.raw_bit_depth) - 1;

    return kernel_params;
}

}  // namespace

CcmBlock::CcmBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

__global__ void
ccm_kernel(FrameView<uint16_t> in, FrameView<uint16_t> out, CcmKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    const int32_t r = in.at(y, x, 0);
    const int32_t g = in.at(y, x, 1);
    const int32_t b = in.at(y, x, 2);

    const int32_t sat = static_cast<int32_t>(params.sat_value);

#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        // TODO:
        //  Drop this once the ISP pipeline is verified
#ifdef USE_CCM_BIAS
        const int32_t corrected = (r * params.matrix[c][0] + g * params.matrix[c][1] +
                                   b * params.matrix[c][2] + params.bias[c]) >>
                                  10;
#else
        const int32_t corrected =
            (r * params.matrix[c][0] + g * params.matrix[c][1] + b * params.matrix[c][2]) >> 10;
#endif
        out.at(y, x, c) = static_cast<uint16_t>(max(0, min(corrected, sat)));
    }
}

void
CcmBlock::execute(PipelineData& data)
{
    if (!data.rgb_hdr)
        throw std::runtime_error("CcmBlock: rgb_hdr frame is null");

    auto& frame = *data.rgb_hdr;
    auto out =
        std::make_unique<PitchedFrame<uint16_t>>(frame.width(), frame.height(), frame.channels());

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    ccm_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.rgb_hdr = std::move(out);
}
