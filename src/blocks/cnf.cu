// File: cnf.cu
// Description: CNF CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <cctype>
#include <memory>
#include <stdexcept>

#include "blocks/cnf.hpp"

namespace
{

CnfKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("cnf");
    if (!t)
        throw std::runtime_error("CnfBlock: missing [cnf] config section");

    auto get_int = [&](const char* key) -> int32_t
    { return static_cast<int32_t>((*t)[key].value<int64_t>().value_or(0)); };

    CnfKernelParams kernel_params;
    kernel_params.diff_threshold = get_int("diff_threshold");
    kernel_params.r_gain = get_int("r_gain");
    kernel_params.b_gain = get_int("b_gain");
    kernel_params.sat_value = (1u << cfg.hardware.raw_bit_depth) - 1;
    kernel_params.luma_shift =
        static_cast<uint8_t>(cfg.hardware.raw_bit_depth > 8 ? cfg.hardware.raw_bit_depth - 8 : 0);

    struct Pos
    {
        uint8_t row, col;
    };

    // clang-format off
    static constexpr Pos positions[4] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    // clang-format on

    const std::string& pat = cfg.hardware.bayer_pattern;
    if (pat.size() != 4)
        throw std::runtime_error("CnfBlock: bayer_pattern must be 4 chars");

    Pos r_pos{}, b_pos{};
    bool found_r = false, found_b = false;
    for (int i = 0; i < 4; ++i)
    {
        char c = std::tolower(pat[i]);
        if (c == 'r')
        {
            r_pos = positions[i];
            found_r = true;
        }
        if (c == 'b')
        {
            b_pos = positions[i];
            found_b = true;
        }
    }
    if (!found_r || !found_b)
        throw std::runtime_error("CnfBlock: invalid bayer_pattern '" + pat + "'");

    kernel_params.r_row = r_pos.row;
    kernel_params.r_col = r_pos.col;
    kernel_params.b_row = b_pos.row;
    kernel_params.b_col = b_pos.col;

    for (int i = 0; i < 4; ++i)
    {
        if (std::tolower(pat[i]) == 'g')
        {
            if (positions[i].row == r_pos.row)
            {
                kernel_params.gr_row = positions[i].row;
                kernel_params.gr_col = positions[i].col;
            }
            else
            {
                kernel_params.gb_row = positions[i].row;
                kernel_params.gb_col = positions[i].col;
            }
        }
    }

    return kernel_params;
}

}  // namespace

CnfBlock::CnfBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

/*
    The fade curves are ported from the openIsp but expressed in Q8 to match the project nature
*/
__device__ __forceinline__ int32_t
fade1_q8(int32_t y8)
{
    if (y8 <= 30)
        return 256;
    if (y8 <= 50)
        return 230;
    if (y8 <= 70)
        return 205;
    if (y8 <= 100)
        return 179;
    if (y8 <= 150)
        return 154;
    if (y8 <= 200)
        return 77;
    if (y8 <= 250)
        return 26;
    return 0;
}

__device__ __forceinline__ int32_t
fade2_q8(int32_t avg_c1_8)
{
    if (avg_c1_8 <= 30)
        return 256;
    if (avg_c1_8 <= 50)
        return 230;
    if (avg_c1_8 <= 70)
        return 205;
    if (avg_c1_8 <= 100)
        return 154;
    if (avg_c1_8 <= 150)
        return 128;
    if (avg_c1_8 <= 200)
        return 77;
    return 0;
}

/*
   avg_g -> 5x5 green avg
   avg_same_channel -> 5x5 avg of the same channel of the noisy pixel
   avg_opp_channel -> 5x5 avg of the other channel
   luma -> luma estimate shifted down to 8-bit internally
   luma_shift -> bits to shift luma/avg to 8-bit range
*/
__device__ __forceinline__ int32_t
cn_correct(int32_t noisy_pix_value, int32_t avg_g, int32_t avg_same_channel,
           int32_t avg_opp_channel, int32_t luma, int32_t awb_gain, int32_t luma_shift)
{
    // the higher the gain -> the higher the channel is boosted -> the higher the noise component
    // -> give it a smaller damp_factor to pull it toward the avg
    const int32_t damp = (awb_gain <= 1024) ? 256 : (awb_gain <= 1229) ? 128 : 77;

    const int32_t max_avg = max(avg_g, avg_opp_channel);
    const int32_t signal_gap = noisy_pix_value - max_avg;
    const int32_t chroma_corr = max_avg + ((damp * signal_gap) >> 8);

    // fade: back off correction in bright / high-channel-value areas
    const int32_t f1 = fade1_q8(luma >> luma_shift);
    const int32_t f2 = fade2_q8(avg_same_channel >> luma_shift);
    const int32_t f = (f1 * f2) >> 8;  // combined Q8 fade

    return (f * chroma_corr + (256 - f) * noisy_pix_value) >> 8;
}

__global__ void
cnf_kernel(FrameView<uint16_t> in, FrameView<uint16_t> out, CnfKernelParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    const int c_row = (int)(y & 1);
    const int c_col = (int)(x & 1);

    const bool is_r = (c_row == params.r_row) && (c_col == params.r_col);
    const bool is_b = (c_row == params.b_row) && (c_col == params.b_col);

    // if its nor R/B (if its Green), do nothing
    if (!is_r && !is_b)
    {
        out.at(y, x) = in.at(y, x);
        return;
    }

    // This is a question of (how many rows/cols to shift to get a pixel of that color)
    // Inputs are always {0, 1} so the output is {-1, 0, 1}, we will use it as a guide to move us to
    // the next pixel
    const int r_dy = (int)params.r_row - c_row;
    const int r_dx = (int)params.r_col - c_col;
    const int gr_dy = (int)params.gr_row - c_row;
    const int gr_dx = (int)params.gr_col - c_col;
    const int gb_dy = (int)params.gb_row - c_row;
    const int gb_dx = (int)params.gb_col - c_col;
    const int b_dy = (int)params.b_row - c_row;
    const int b_dx = (int)params.b_col - c_col;

    // Accumulate 5x5 same-channel neighborhoods (9x9 Bayer footprint, step=2)
    int32_t sum_r = 0, sum_gr = 0, sum_gb = 0, sum_b = 0;

    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            const int h = (int)in.height - 1;
            const int w = (int)in.width - 1;

            // To prevent the coordinates from going to negative or beyond in.height, min, max are
            // used for clamping

            // ny_r -> red neighbor in the y direction, and so on
            const uint32_t ny_r = (uint32_t)max(0, min(h, (int)y + r_dy + 2 * dy));
            const uint32_t nx_r = (uint32_t)max(0, min(w, (int)x + r_dx + 2 * dx));
            const uint32_t ny_gr = (uint32_t)max(0, min(h, (int)y + gr_dy + 2 * dy));
            const uint32_t nx_gr = (uint32_t)max(0, min(w, (int)x + gr_dx + 2 * dx));
            const uint32_t ny_gb = (uint32_t)max(0, min(h, (int)y + gb_dy + 2 * dy));
            const uint32_t nx_gb = (uint32_t)max(0, min(w, (int)x + gb_dx + 2 * dx));
            const uint32_t ny_b = (uint32_t)max(0, min(h, (int)y + b_dy + 2 * dy));
            const uint32_t nx_b = (uint32_t)max(0, min(w, (int)x + b_dx + 2 * dx));

            sum_r += in.at(ny_r, nx_r);
            sum_gr += in.at(ny_gr, nx_gr);
            sum_gb += in.at(ny_gb, nx_gb);
            sum_b += in.at(ny_b, nx_b);
        }
    }

    const int32_t avg_r = sum_r / 25;
    const int32_t avg_gr = sum_gr / 25;
    const int32_t avg_gb = sum_gb / 25;
    const int32_t avg_b = sum_b / 25;
    const int32_t avg_g = (avg_gr + avg_gb) >> 1;

    const int32_t luma = (306 * avg_r + 601 * avg_g + 117 * avg_b) >> 10;

    const int32_t curr = in.at(y, x);
    const int32_t diff_threshold = params.diff_threshold;
    int32_t result = curr;

    /*
       If a pixel
        - Spikes above the local green values ((curr - avg_g) > diff_threshold)
        - Spikes above the local blue values ((curr - avg_b) > diff_threshold)
        - The whole local Red is higher compared to green but Red is not high compared to Blue
       Its a noisy pixel
    */
    if (is_r)
    {
        const bool is_noise = (curr - avg_g > diff_threshold) && (curr - avg_b > diff_threshold) &&
                              (avg_r - avg_g > diff_threshold) && (avg_r - avg_b < diff_threshold);
        if (is_noise)
            result = cn_correct(curr, avg_g, avg_r, avg_b, luma, params.r_gain, params.luma_shift);
    }
    else  // is_b
    {
        const bool is_noise = (curr - avg_g > diff_threshold) && (curr - avg_r > diff_threshold) &&
                              (avg_b - avg_g > diff_threshold) && (avg_b - avg_r < diff_threshold);
        if (is_noise)
            result = cn_correct(curr, avg_g, avg_b, avg_r, luma, params.b_gain, params.luma_shift);
    }

    const int32_t sat = (int32_t)params.sat_value;
    out.at(y, x) = static_cast<uint16_t>(max(0, min(result, sat)));
}

void
CnfBlock::execute(PipelineData& data)
{
    if (!data.bayer)
        throw std::runtime_error("CnfBlock: bayer frame is null");

    auto& frame = *data.bayer;
    auto out =
        std::make_unique<PitchedFrame<uint16_t>>(frame.width(), frame.height(), frame.channels());

    const dim3 block(16, 16);
    const dim3 grid((frame.width() + block.x - 1) / block.x,
                    (frame.height() + block.y - 1) / block.y);

    cnf_kernel<<<grid, block, 0, stream_>>>(frame.view(), out->view(), params_);

    data.bayer = std::move(out);
}
