// File: blc.cu
// Description: BLC CUDA kernel and block implementation
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <cctype>
#include <stdexcept>

#include <toml++/toml.hpp>

#include "blc.hpp"

namespace
{

/*
   This function will
   1- parse the block configurations
    - Figure out the bayer order (only supports rggb, gbbr)
    - Construct a 2x2 tile with known positions so the kernel knows which pixel is which
*/

BlcKernelParams
make_params(const IspConfig& cfg)
{
    const toml::table* t = cfg.block_params("blc");
    if (!t)
    {
        throw std::runtime_error("BlcBlock: missing [blc] config section");
    }

    auto get_int = [&](const char* key) -> int32_t
    { return static_cast<int32_t>((*t)[key].value<int64_t>().value_or(0)); };

    BlcKernelParams kernel_params{};
    kernel_params.bl_r = get_int("bl_r");
    kernel_params.bl_gr = get_int("bl_gr");
    kernel_params.bl_gb = get_int("bl_gb");
    kernel_params.bl_b = get_int("bl_b");
    kernel_params.alpha = get_int("alpha");
    kernel_params.beta = get_int("beta");
    kernel_params.sat_value = static_cast<uint32_t>((1u << cfg.hardware.raw_bit_depth) - 1);

    struct Pos
    {
        uint8_t row, col;
    };

    // clang-format off
    static constexpr Pos positions[4] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    // clang-format on

    const std::string& pat = cfg.hardware.bayer_pattern;
    if (pat.size() != 4)
    {
        throw std::runtime_error("BlcBlock: bayer_pattern must be 4 characters, got '" + pat + "'");
    }

    Pos r_pos{}, gr_pos{}, gb_pos{}, b_pos{};
    bool found_r = false, found_b = false;

    for (int i = 0; i < 4; ++i)
    {
        char c = std::tolower(pat[i]);
        if (c == 'r')
        {
            r_pos = positions[i];
            found_r = true;
        }
        else if (c == 'b')
        {
            b_pos = positions[i];
            found_b = true;
        }
    }
    // if its not bggr or rggb, PANIC!!
    if (!found_r || !found_b)
    {
        throw std::runtime_error("BlcBlock: invalid bayer_pattern '" + pat + "'");
    }

    for (int i = 0; i < 4; ++i)
    {
        char c = std::tolower(pat[i]);
        if (c == 'g')
        {
            if (positions[i].row == r_pos.row)
            {
                gr_pos = positions[i];
            }
            else
            {
                gb_pos = positions[i];
            }
        }
    }

    kernel_params.r_row = r_pos.row;
    kernel_params.r_col = r_pos.col;
    kernel_params.gr_row = gr_pos.row;
    kernel_params.gr_col = gr_pos.col;
    kernel_params.gb_row = gb_pos.row;
    kernel_params.gb_col = gb_pos.col;
    kernel_params.b_row = b_pos.row;
    kernel_params.b_col = b_pos.col;

    return kernel_params;
}

}  // namespace

/*
   Each thread will process a 2x2 tile since Gr correction depends on the post-subtraction
   of the R value, and Gb on B. So I will process them all in one thread
   to make sure that im reading all of them before any writing (by another thread)
*/
__global__ void
blc_kernel(FrameView<uint16_t> frame, BlcKernelParams params)
{
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    if (x + 1 >= frame.width || y + 1 >= frame.height)
        return;

    int32_t r = frame.at(y + params.r_row, x + params.r_col);
    int32_t gr = frame.at(y + params.gr_row, x + params.gr_col);
    int32_t gb = frame.at(y + params.gb_row, x + params.gb_col);
    int32_t b = frame.at(y + params.b_row, x + params.b_col);

    r = max(r - params.bl_r, 0);
    b = max(b - params.bl_b, 0);
    gr = gr - params.bl_gr + (r * params.alpha >> 10);
    gb = gb - params.bl_gb + (b * params.beta >> 10);

    const int32_t sat = params.sat_value;
    auto clamp16 = [sat](int32_t v) -> uint16_t
    { return static_cast<uint16_t>(max(min(v, sat), 0)); };

    frame.at(y + params.r_row, x + params.r_col) = clamp16(r);
    frame.at(y + params.gr_row, x + params.gr_col) = clamp16(gr);
    frame.at(y + params.gb_row, x + params.gb_col) = clamp16(gb);
    frame.at(y + params.b_row, x + params.b_col) = clamp16(b);
}

BlcBlock::BlcBlock(const IspConfig& cfg, cudaStream_t stream)
    : IspBlock(cfg, stream), params_(make_params(cfg))
{
}

void
BlcBlock::execute(PipelineData& data)
{
    if (!data.bayer)
        throw std::runtime_error("BlcBlock: bayer frame is null");

    auto& frame = *data.bayer;
    const dim3 block(16, 16);
    const dim3 grid((frame.width() / 2 + block.x - 1) / block.x,
                    (frame.height() / 2 + block.y - 1) / block.y);

    blc_kernel<<<grid, block, 0, stream_>>>(frame.view(), params_);
}
