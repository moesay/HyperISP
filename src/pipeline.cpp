// File: pipeline.cpp
// Description: Built-in block wiring and pipeline execution
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#include <algorithm>
#include <stdexcept>

#include <pipeline.hpp>

#include "blocks/aaf.hpp"
#include "blocks/awb.hpp"
#include "blocks/blc.hpp"
#include "blocks/bnf.hpp"
#include "blocks/ccm.hpp"
#include "blocks/cfa.hpp"
#include "blocks/cnf.hpp"
#include "blocks/csc.hpp"
#include "blocks/dpc.hpp"
#include "blocks/gac.hpp"
#include "blocks/nlm.hpp"

IspPipeline::IspPipeline(const IspConfig& cfg, cudaStream_t stream) : stream_(stream)
{
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);

    auto add_if_enabled = [&](std::unique_ptr<IspBlock> block, const std::string& dependency = "")
    {
        auto it = cfg.block_enable_status.find(block->name());
        if (it == cfg.block_enable_status.end() || !it->second)
            return;

        if (!dependency.empty())
        {
            const bool dependency_present =
                std::any_of(blocks_.begin(), blocks_.end(),
                            [&](const auto& b) { return b->name() == dependency; });
            if (!dependency_present)
                return;
        }

        blocks_.push_back(std::move(block));
    };

    add_if_enabled(std::make_unique<DpcBlock>(cfg, stream));
    add_if_enabled(std::make_unique<BlcBlock>(cfg, stream));
    add_if_enabled(std::make_unique<AafBlock>(cfg, stream));
    add_if_enabled(std::make_unique<AwbBlock>(cfg, stream));
    add_if_enabled(std::make_unique<CnfBlock>(cfg, stream));
    add_if_enabled(std::make_unique<CfaBlock>(cfg, stream));
    add_if_enabled(std::make_unique<CcmBlock>(cfg, stream));
    add_if_enabled(std::make_unique<GacBlock>(cfg, stream));
    add_if_enabled(std::make_unique<CscBlock>(cfg, stream));
    add_if_enabled(std::make_unique<NlmBlock>(cfg, stream), "csc");
    add_if_enabled(std::make_unique<BnfBlock>(cfg, stream), "csc");
}

IspPipeline::~IspPipeline()
{
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void
IspPipeline::append_block(std::unique_ptr<IspBlock> block)
{
    blocks_.push_back(std::move(block));
}

void
IspPipeline::insert_block_after(const std::string& name, std::unique_ptr<IspBlock> block)
{
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
                           [&](const auto& b) { return b->name() == name; });
    if (it == blocks_.end())
    {
        throw std::runtime_error("IspPipeline::insert_block_after: no block named '" + name +
                                 "' in pipeline");
    }
    blocks_.insert(it + 1, std::move(block));
}

void
IspPipeline::process(PipelineData& data)
{
    for (auto& block : blocks_)
    {
        cudaEventRecord(start_, stream_);
        block->execute(data);
        cudaEventRecord(stop_, stream_);
        cudaStreamSynchronize(stream_);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("IspPipeline: CUDA error after [") +
                                     block->name() + "]: " + cudaGetErrorString(err));
        }

        if (stage_callback_)
        {
            float elapsed_ms = 0.0f;
            cudaEventElapsedTime(&elapsed_ms, start_, stop_);
            stage_callback_(block->name(), elapsed_ms);
        }
    }
}

std::vector<std::string>
IspPipeline::block_order() const
{
    std::vector<std::string> names;
    names.reserve(blocks_.size());
    for (auto& block : blocks_)
        names.push_back(block->name());
    return names;
}

void
IspPipeline::set_stage_callback(StageCallback cb)
{
    stage_callback_ = std::move(cb);
}
