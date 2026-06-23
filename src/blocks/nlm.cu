// File: nlm.cu
// Description: NLM CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "blocks/nlm.hpp"

#define NLM_MAX_PATCH 15

namespace
{

constexpr int32_t kLutSize = 255 * 255;

NlmKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("nlm");
    if (!t)
    {
        throw std::runtime_error("NlmBlock: missing [nlm] config section");
    }

    auto get_int = [&](const char* key) -> int32_t
    { return static_cast<int32_t>((*t)[key].value<int64_t>().value_or(0)); };

    NlmKernelParams kernel_params{};
    kernel_params.search_window_size = get_int("search_window_size");
    kernel_params.patch_size = get_int("patch_size");
    kernel_params.lut_size = kLutSize;

    if (kernel_params.search_window_size <= 0 || kernel_params.search_window_size % 2 == 0)
    {
        throw std::runtime_error("NlmBlock: search_window_size must be a positive odd integer");
    }

    if (kernel_params.patch_size <= 0 || kernel_params.patch_size % 2 == 0 ||
        kernel_params.patch_size > NLM_MAX_PATCH)
    {
        throw std::runtime_error("NlmBlock: patch_size must be a positive odd integer <= " +
                                 std::to_string(NLM_MAX_PATCH));
    }

    const double h = static_cast<double>(get_int("h"));
    std::vector<int32_t> host_lut(kLutSize);

    for (int32_t d = 0; d < kLutSize; ++d)
    {
        host_lut[d] = static_cast<int32_t>(1024.0 * std::exp(-static_cast<double>(d) / (h * h)));
    }

    int32_t* device_lut = nullptr;
    cudaError_t err = cudaMalloc(&device_lut, kLutSize * sizeof(int32_t));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("NlmBlock: cudaMalloc failed: ") +
                                 cudaGetErrorString(err));
    }

    err =
        cudaMemcpy(device_lut, host_lut.data(), kLutSize * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(device_lut);
        throw std::runtime_error(std::string("NlmBlock: cudaMemcpy failed: ") +
                                 cudaGetErrorString(err));
    }

    kernel_params.weights_lut = device_lut;
    return kernel_params;
}

__device__ __forceinline__ int
reflect_index(int idx, int size)
{
    if (idx < 0)
        return -idx;
    if (idx >= size)
        return 2 * size - idx - 2;
    return idx;
}

}  // namespace

/*
   The mental model:
   - For a pixel at (y, x), construct a patch of its own, slide over each other in the search
   window around p(y, x)
   - Doing this will make you land on a pixel, a candidate pixel
   - For each candidate pixel, build its patch too and compute the ssd and compare it against
   the center patch, this will give you the distance
   - Convert this distance into a weight using exp(-distance / h^2) { small distance -> batches
   are alike and vice versa }
   - Only the luma channel is denoised, the rest are passed through untouched.

   PS: Since param in is never written in the kernel, we can safely use read-only-cache hint __ldg
*/

__global__ void
nlm_kernel(FrameView<uint8_t> in, FrameView<uint8_t> out, NlmKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    out.at(y, x, 1) = __ldg(&in.at(y, x, 1));
    out.at(y, x, 2) = __ldg(&in.at(y, x, 2));

    const int height = static_cast<int>(in.height);
    const int width = static_cast<int>(in.width);

    const int search_r = params.search_window_size / 2;
    const int patch_r = params.patch_size / 2;
    const int patch_area = params.patch_size * params.patch_size;

    // reflected coordinates of the current pixel
    int reflected_y[NLM_MAX_PATCH];
    int reflected_x[NLM_MAX_PATCH];
    int32_t center_patch[NLM_MAX_PATCH][NLM_MAX_PATCH];

    /*
       Try to visualize this by a pin and a paper if its not convenient, I will try to draw it below
       too.
       X -> Patch center

       ----------X----------

       Iterating from 0 to patch_size and subtracting the patch center each time will give us the
       points that builds the patch. It works for X and Y directions.
       This is the technique used to traverse the window here and in the search nested loops

    */
    for (int i = 0; i < params.patch_size; ++i)
    {
        reflected_y[i] = reflect_index(static_cast<int>(y) + i - patch_r, height);
        reflected_x[i] = reflect_index(static_cast<int>(x) + i - patch_r, width);
    }

    // After we have all the patch points, build the patch
    for (int i = 0; i < params.patch_size; ++i)
        for (int j = 0; j < params.patch_size; ++j)
            center_patch[i][j] = __ldg(&in.at(static_cast<uint32_t>(reflected_y[i]),
                                              static_cast<uint32_t>(reflected_x[j]), 0));

    int64_t weighted_sum = 0;
    int64_t weight_total = 0;

    // per-search offset arrays
    int sy[NLM_MAX_PATCH];
    int sx[NLM_MAX_PATCH];

    for (int dy = -search_r; dy <= search_r; ++dy)
    {
        for (int i = 0; i < params.patch_size; ++i)
            sy[i] = reflect_index(reflected_y[i] + dy, height);

        for (int dx = -search_r; dx <= search_r; ++dx)
        {
            for (int j = 0; j < params.patch_size; ++j)
                sx[j] = reflect_index(reflected_x[j] + dx, width);

            int64_t patch_sum_squared_diff = 0;
            for (int i = 0; i < params.patch_size; ++i)
            {
                const uint8_t* row = in.row_ptr(static_cast<uint32_t>(sy[i]));
                for (int j = 0; j < params.patch_size; ++j)
                {
                    const int32_t diff =
                        center_patch[i][j] - static_cast<int32_t>(__ldg(row + sx[j] * in.channels));
                    patch_sum_squared_diff += diff * diff;
                }
            }

            const int32_t distance =
                min(static_cast<int32_t>(patch_sum_squared_diff / patch_area), params.lut_size - 1);
            const int32_t weight = __ldg(params.weights_lut + distance);

            weighted_sum +=
                static_cast<int64_t>(weight) *
                __ldg(in.row_ptr(static_cast<uint32_t>(sy[patch_r])) + sx[patch_r] * in.channels);
            weight_total += weight;
        }
    }

    const int32_t result = static_cast<int32_t>(weighted_sum / weight_total);
    out.at(y, x, 0) = static_cast<uint8_t>(max(0, min(result, 255)));
}

NlmBlock::NlmBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

NlmBlock::~NlmBlock()
{
    if (params_.weights_lut)
        cudaFree(const_cast<int32_t*>(params_.weights_lut));
}

void
NlmBlock::execute(PipelineData& data)
{
    if (!data.ycbcr)
        throw std::runtime_error("NlmBlock: ycbcr frame is null");

    auto& frame = *data.ycbcr;
    auto out =
        std::make_unique<PitchedFrame<uint8_t>>(frame.width(), frame.height(), frame.channels());

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    nlm_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.ycbcr = std::move(out);
}
