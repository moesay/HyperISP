// File: bnf.cu
// Description: BNF CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <toml++/toml.hpp>

#include "bnf.hpp"

namespace
{

BnfKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("bnf");
    if (!t)
    {
        throw std::runtime_error("BnfBlock: missing [bnf] config section");
    }

    auto get_int = [&](const char* key) -> int32_t
    { return static_cast<int32_t>((*t)[key].value<int64_t>().value_or(0)); };

    auto get_float = [&](const char* key) -> float
    { return static_cast<float>((*t)[key].value<double>().value_or(0.0)); };

    BnfKernelParams kernel_params{};

    kernel_params.intensity_sigma = get_float("intensity_sigma");
    kernel_params.spatial_sigma = get_float("spatial_sigma");
    kernel_params.kernel_size = get_int("kernel_size");

    if (kernel_params.kernel_size <= 0 || kernel_params.kernel_size % 2 == 0)
    {
        throw std::runtime_error("BnfBlock: kernel_size must be a positive odd integer");
    }

    // max squared spatial distance from center to any corner of the kernel = 2 * radius^2
    // radius = (k-1)/2, so max_dist_sq = lut_size = (k-1)^2/2
    const uint32_t lut_size = static_cast<uint32_t>((kernel_params.kernel_size - 1) *
                                                    (kernel_params.kernel_size - 1) / 2) +
                              1;
    kernel_params.lut_size = lut_size;

    std::vector<int32_t> host_lut(lut_size);

    const double inv_2_sigma_s_sq =
        1.0 / (2.0 * kernel_params.spatial_sigma * kernel_params.spatial_sigma);
    for (int32_t d = 0; d < static_cast<int32_t>(lut_size); ++d)
    {
        // Q10 scaling by 1024 to make the range varies from 0 to 1024
        host_lut[d] =
            static_cast<int32_t>(1024.0 * std::exp(-static_cast<double>(d) * inv_2_sigma_s_sq));
    }

    int32_t* device_lut = nullptr;
    cudaError_t err = cudaMalloc(&device_lut, lut_size * sizeof(int32_t));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("BnfBlock: cudaMalloc failed: ") +
                                 cudaGetErrorString(err));
    }

    err =
        cudaMemcpy(device_lut, host_lut.data(), lut_size * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(device_lut);
        throw std::runtime_error(std::string("BnfBlock: cudaMemcpy failed: ") +
                                 cudaGetErrorString(err));
    }

    kernel_params.spatial_weights_lut = device_lut;
    return kernel_params;
}

}  // namespace

BnfBlock::BnfBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

BnfBlock::~BnfBlock()
{
    if (params_.spatial_weights_lut)
        cudaFree(const_cast<int32_t*>(params_.spatial_weights_lut));
}

__global__ void
bnf_kernel(FrameView<uint8_t> in, FrameView<uint8_t> out, BnfKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    out.at(y, x, 1) = __ldg(&in.at(y, x, 1));
    out.at(y, x, 2) = __ldg(&in.at(y, x, 2));

    const int height = static_cast<int>(in.height);
    const int width = static_cast<int>(in.width);
    const int radius = params.kernel_size / 2;
    const int center_y = static_cast<int>(y);
    const int center_x = static_cast<int>(x);
    const int center_val = static_cast<int>(__ldg(&in.at(y, x, 0)));

    const float inv_2_sigma_i_sq = 1.0f / (2.0f * params.intensity_sigma * params.intensity_sigma);

    int64_t weighted_sum = 0;
    int64_t weight_total = 0;

    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int ny = max(0, min(height - 1, center_y + dy));

        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int nx = max(0, min(width - 1, center_x + dx));

            const int d_sq = dy * dy + dx * dx;
            const int32_t w_s = __ldg(params.spatial_weights_lut +
                                      min(d_sq, static_cast<int>(params.lut_size) - 1));

            const int neighbor_val = static_cast<int>(
                __ldg(&in.at(static_cast<uint32_t>(ny), static_cast<uint32_t>(nx), 0)));
            const int delta_i = center_val - neighbor_val;
            const int32_t w_i = static_cast<int32_t>(
                1024.0f * __expf(-static_cast<float>(delta_i * delta_i) * inv_2_sigma_i_sq));

            // Q10 * Q10 gives Q20 fixed point weights
            const int64_t combined_weight = static_cast<int64_t>(w_s) * w_i;
            weighted_sum += combined_weight * neighbor_val;
            weight_total += combined_weight;
        }
    }

    const int32_t result =
        (weight_total > 0) ? static_cast<int32_t>(weighted_sum / weight_total) : center_val;
    out.at(y, x, 0) = static_cast<uint8_t>(max(0, min(result, 255)));
}

void
BnfBlock::execute(PipelineData& data)
{
    if (!data.ycbcr)
    {
        throw std::runtime_error("BnfBlock: ycbcr frame is null");
    }

    auto& frame = *data.ycbcr;
    auto out =
        std::make_unique<PitchedFrame<uint8_t>>(frame.width(), frame.height(), frame.channels());

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    bnf_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.ycbcr = std::move(out);
}
