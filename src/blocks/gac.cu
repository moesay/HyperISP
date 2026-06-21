// File: gac.cu
// Description: GAC CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <stdexcept>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "blocks/gac.hpp"

namespace
{

constexpr uint32_t SDR_SAT = 255;

struct GammaLut
{
    double gamma;
    double hdr_sat;

    __host__ __device__ uint8_t
    operator()(uint32_t x) const
    {
        // lut[x] = (x / hdr_sat) ^ (gamma * 255)
        return static_cast<uint8_t>(pow(static_cast<double>(x) / hdr_sat, gamma) * SDR_SAT);
    }
};

GacKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("gac");
    if (!t)
        throw std::runtime_error("GacBlock: missing [gac] config section");

    GacKernelParams kernel_params{};
    kernel_params.gain = static_cast<uint32_t>((*t)["gain"].value<int64_t>().value_or(256));
    kernel_params.hdr_sat = (1u << cfg.hardware.raw_bit_depth) - 1;

    const double gamma = (*t)["gamma"].value<double>().value_or(1.0);
    const uint32_t lut_size = kernel_params.hdr_sat + 1;

    uint8_t* device_lut = nullptr;
    cudaError_t err = cudaMalloc(&device_lut, lut_size * sizeof(uint8_t));

    if (err != cudaSuccess)
        throw std::runtime_error(std::string("GacBlock: cudaMalloc failed: ") +
                                 cudaGetErrorString(err));

    // Precompute the gamma lut on the gpu, store it on the device memory to maximize speed
    thrust::transform(thrust::device, thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(lut_size), device_lut,
                      GammaLut{ gamma, static_cast<double>(kernel_params.hdr_sat) });

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        cudaFree(device_lut);
        throw std::runtime_error(std::string("GacBlock: gamma LUT build failed: ") +
                                 cudaGetErrorString(err));
    }

    kernel_params.lut = device_lut;

    return kernel_params;
}

}  // namespace

GacBlock::GacBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

GacBlock::~GacBlock()
{
    if (params_.lut)
        cudaFree(const_cast<uint8_t*>(params_.lut));
}

__global__ void
gac_kernel(FrameView<uint16_t> in, FrameView<uint8_t> out, GacKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    const int32_t hdr_sat = static_cast<int32_t>(params.hdr_sat);

#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        // Apply the digital gain but still, hdr
        const int32_t gained = (static_cast<int32_t>(params.gain) * in.at(y, x, c)) >> 8;
        const int32_t clamped = max(0, min(gained, hdr_sat));
        // sdr, 8-bit
        out.at(y, x, c) = params.lut[clamped];
    }
}

void
GacBlock::execute(PipelineData& data)
{
    if (!data.rgb_hdr)
        throw std::runtime_error("GacBlock: rgb_hdr frame is null");

    auto& frame = *data.rgb_hdr;
    auto out =
        std::make_unique<PitchedFrame<uint8_t>>(frame.width(), frame.height(), frame.channels());

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    gac_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.rgb_sdr = std::move(out);
}
