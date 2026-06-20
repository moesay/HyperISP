// File: main.cpp
// Description: ISP pipeline entry point
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <cstdlib>
#include <memory>
#include <print>
#include <vector>

#include <blocks/aaf.hpp>
#include <blocks/awb.hpp>
#include <blocks/blc.hpp>
#include <blocks/cfa.hpp>
#include <blocks/cnf.hpp>
#include <blocks/dpc.hpp>
#include <config.hpp>
#include <frame.hpp>
#include <frames_io.hpp>
#include <isp_block.hpp>
#include <pipeline_data.hpp>

static std::vector<std::unique_ptr<IspBlock>>
build_pipeline(const IspConfig& cfg, cudaStream_t stream = 0)
{
    std::vector<std::unique_ptr<IspBlock>> pipeline;

    auto add_if_enabled = [&](std::unique_ptr<IspBlock> block)
    {
        auto it = cfg.block_enable_status.find(block->name());
        if (it != cfg.block_enable_status.end() && it->second)
        {
            pipeline.push_back(std::move(block));
        }
    };

    add_if_enabled(std::make_unique<DpcBlock>(cfg, stream));
    add_if_enabled(std::make_unique<BlcBlock>(cfg, stream));
    add_if_enabled(std::make_unique<AafBlock>(cfg, stream));
    add_if_enabled(std::make_unique<AwbBlock>(cfg, stream));
    add_if_enabled(std::make_unique<CnfBlock>(cfg, stream));
    add_if_enabled(std::make_unique<CfaBlock>(cfg, stream));

    return pipeline;
}

int
main(int argc, char* argv[])
{
    std::string config_path = (argc > 1) ? argv[1] : "configs/nikon_d3200.toml";
    std::string raw_path = "./test_raws/test.raw";

    IspConfig cfg;
    try
    {
        cfg = IspConfig::load(config_path);
    }
    catch (const std::exception& e)
    {
        std::println(stderr, "Error loading config: {}", e.what());
        return EXIT_FAILURE;
    }

    std::print("Config loaded: {}\n", config_path);

    PipelineData data;
    try
    {
        auto raw = load_raw(raw_path, cfg.hardware.raw_width, cfg.hardware.raw_height);
        data.bayer = std::make_unique<PitchedFrame<uint16_t>>(std::move(raw));
    }
    catch (const std::exception& e)
    {
        std::println(stderr, "Error loading RAW: {}", e.what());
        return EXIT_FAILURE;
    }

    auto& bayer = *data.bayer;
    const auto& hw = cfg.hardware;
    std::print("RAW loaded   : {}\n", raw_path);
    std::print("  Size       : {}x{}  pitch={} bytes\n", bayer.width(), bayer.height(),
               bayer.pitch());
    std::print("  Sensor     : {}x{}  {}-bit  pattern={}\n", hw.raw_width, hw.raw_height,
               hw.raw_bit_depth, hw.bayer_pattern);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto pipeline = build_pipeline(cfg, stream);

    std::print("Pipeline     :");
    for (auto& block : pipeline)
        std::print(" {}", block->name());
    std::print("\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto& block : pipeline)
    {
        cudaEventRecord(start, stream);
        block->execute(data);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::println(stderr, "CUDA error after [%s]: {}", block->name(),
                         cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
            return EXIT_FAILURE;
        }

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        std::print("  [{}] done in {:.6f} ms\n", block->name(), elapsed_ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    save_raw(*data.bayer, "./output/bayer_out.raw");

    if (data.rgb_hdr)
    {
        save_rgb(*data.rgb_hdr, "./output/rgb_out.raw");
    }

    return EXIT_SUCCESS;
}
