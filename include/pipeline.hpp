// File: pipeline.hpp
// Description: Builds and runs the ordered sequence of ISP blocks
// Author: Mohamed ElKafafy (m.elsayed4420@gmail.com)
// Licensed under the GNU General Public License v3.0 (GPL-3.0)

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "config.hpp"
#include "isp_block.hpp"
#include "pipeline_data.hpp"

class IspPipeline
{
  public:
    using StageCallback = std::function<void(const char* name, float elapsed_ms)>;

    explicit IspPipeline(const IspConfig& cfg, cudaStream_t stream = 0);
    ~IspPipeline();

    IspPipeline(const IspPipeline&) = delete;
    IspPipeline& operator=(const IspPipeline&) = delete;

    void append_block(std::unique_ptr<IspBlock> block);
    void insert_block_after(const std::string& name, std::unique_ptr<IspBlock> block);
    void process(PipelineData& data);
    std::vector<std::string> block_order() const;
    void set_stage_callback(StageCallback cb);

  private:
    std::vector<std::unique_ptr<IspBlock>> blocks_;
    StageCallback stage_callback_;
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
