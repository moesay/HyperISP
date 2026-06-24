// File: main.cpp
// Description: ISP pipeline entry point
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <cstdlib>
#include <filesystem>
#include <print>

#include <cmdparser.hpp>
#include <frames_io.hpp>
#include <hyperisp.hpp>

int
main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>("c", "config", "../configs/nikon_d3200.toml",
                                     "Path to the ISP config TOML file");
    parser.set_optional<std::string>("r", "raw", "../test_raws/test.raw",
                                     "Path to the input RAW file");
    parser.run_and_exit_if_error();

    std::string config_path = parser.get<std::string>("c");
    std::string raw_path = parser.get<std::string>("r");

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

    IspPipeline pipeline(cfg, stream);

    std::print("Pipeline     :");
    for (auto& name : pipeline.block_order())
        std::print(" {}", name);
    std::print("\n");

    pipeline.set_stage_callback([](const char* name, float elapsed_ms)
    { std::print("  [{}] done in {:.6f} ms\n", name, elapsed_ms); });

    try
    {
        pipeline.process(data);
    }
    catch (const std::exception& e)
    {
        std::println(stderr, "{}", e.what());
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    cudaStreamDestroy(stream);

    try
    {
        std::filesystem::create_directory("output");
    }
    catch (std::filesystem::filesystem_error& e)
    {
        std::println(stderr, "{}", e.what());
        return EXIT_FAILURE;
    }

    save_raw(*data.bayer, "./output/bayer_out.raw");

    if (data.rgb_hdr)
    {
        save_rgb(*data.rgb_hdr, "./output/rgb_out_hdr.raw");
    }

    if (data.rgb_sdr)
    {
        save_rgb(*data.rgb_sdr, "./output/rgb_out_sdr.raw");
    }

    if (data.ycbcr)
    {
        save_rgb(*data.ycbcr, "./output/ycbcr_out.raw");
    }

    return EXIT_SUCCESS;
}
