<p align="center">
  <a href="https://github.com/moesay/hyperISP/">
    <img src="assets/logo.png" alt="HyperISP" width="600">
  </a>
</p>

<p align="center">
  A GPU-accelerated re-implementation of <a href="https://github.com/QiuJueqin/fast-openISP">fast-openISP</a>,
  aiming to push an image signal processing pipeline onto the GPU for real-time throughput.
  <br>
  <a href="https://github.com/moesay/hyperISP/issues/new">Report bug</a>
  ·
  <a href="https://github.com/moesay/hyperISP/issues/new">Request feature</a>
</p>

<p align="center">
      <a href="https://github.com/moesay/hyperISP/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/moesay/hyperISP" /></a>
      <a href="https://github.com/moesay/hyperISP/" alt="Status">
        <img src="https://img.shields.io/badge/Status-WIP-f10" /></a>
      <a href="https://github.com/moesay/hyperISP/" alt="Dev Status">
        <img src="https://img.shields.io/badge/Developing-Active-green" /></a>
</p>

## Table of contents

- [Why?](#why)
- [Overview](#overview)
  - [Pipeline](#pipeline)
- [Performance](#performance)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Copyright and license](#copyright-and-license)

## Why?

[openISP](https://github.com/cruxopen/openISP) is a Python reference implementation of a classic
image signal processing pipeline. [fast-openISP](https://github.com/QiuJueqin/fast-openISP) is its
successor: a drop-in, NumPy-vectorized rewrite that gets the same pipeline running **over 300x
faster** on the CPU.

HyperISP picks up where fast-openISP leaves off. The pipeline stages are inherently
data-parallel. Each one maps a function over every pixel of a frame, which makes them a natural
fit for the GPU. By re-implementing the pipeline in C++/CUDA, HyperISP aims to take the same
algorithms several steps further: from "fast enough to batch-process on a CPU" to "fast enough
for real-time, frame-by-frame processing".

## Overview

HyperISP is written in modern C++/CUDA. Each pipeline stage is an independent `IspBlock` that
operates on shared `PipelineData` (Bayer, demosaiced RGB, and YCbCr frame buffers), launched on a
CUDA stream. Per-camera tuning is read from a TOML config file (see [`configs/`](configs)), which
also controls which blocks are enabled.

### Pipeline

Mirroring the fast-openISP pipeline, the following stages are planned:

- [x] **DPC** — Dead Pixel Correction
- [x] **BLC** — Black Level Compensation
- [x] **AAF** — Anti-aliasing Filter
- [x] **AWB** — Auto White Balance
- [x] **CFA** — Color Filter Array Demosaicing
- [x] **CNF** — Chroma Noise Filtering
- [x] **CCM** — Color Correction Matrix
- [x] **GAC** — Gamma Correction
- [ ] **CSC** — Color Space Conversion
- [ ] **NLM** — Non-Local Means Denoising
- [ ] **BNF** — Bilateral Noise Filtering
- [ ] **CEH** — Contrast Enhancement
- [ ] **EEH** — Edge Enhancement
- [ ] **FCS** — False Color Suppression
- [ ] **HSC** — Hue & Saturation Control
- [ ] **BCC** — Brightness & Contrast Control
- [ ] **SCL** — Scaling

## Performance

Running times for each stage on a 1920x1080, 12-bit RGGB frame, compared against fast-openISP:

| Block| fast-openISP | HyperISP | Speedup |
|:------:|:-------------------:|:----------------:|:-------:|
| DPC    | 0.29 s                    | 1.9 ms                  | 152.6x         |
| BLC    | 0.02 s                    | 0.02 ms                 | 833.3x         |
| AAF    | 0.08 s                    | 0.9 ms                  | 88.8x         |
| AWB    | 0.02 s                    | 0.03 ms                  | 526.3x         |
| CFA    | 0.20 s                    | 0.94 ms                  | 212.7x         |
| CNF    | 0.25 s                    | 0.7 ms                  | 357.1x         |
| CCM    | 0.06 s                    | 0.7 ms                  | 85.7x         |
| GAC    | 0.07 s                    | 0.4 ms                  | 175x         |
| CSC    | 0.06 s                    |[x]                  |         |
| NLM    | 5.37 s                    |[x]                  |         |
| BNF    | 0.75 s                    |[x]                  |         |
| CEH    | 0.14 s                    |[x]                  |         |
| EEH    | 0.24 s                    |[x]                  |         |
| FCS    | 0.08 s                    |[x]                  |         |
| HSC    | 0.07 s                    |[x]                  |         |
| BCC    | 0.03 s                    |[x]                  |         |
| **End-to-end** | 7.82 s             |[x]                  |         |

## Quick start

**Requirements:**
- CMake 3.20 or higher
- CUDA Toolkit (nvcc 13.3 is preferable) with a C++23-capable host compiler
- Git (for fetching the `tomlplusplus` dependency)

```bash
# Clone the repository
git clone https://github.com/moesay/hyperISP.git
cd hyperISP

# Configure and build
cmake -S . -B build
cmake --build build

# Run on the default config and test RAW
./build/cudaISP configs/nikon_d3200.toml
```

## Configuration

Each camera/sensor has its own TOML config under [`configs/`](configs), describing the sensor
layout (resolution, bit depth, Bayer pattern) and the tunable parameters for every pipeline stage,
plus a per-block enable/disable switch.

## Contributing

Contributions are welcome — feel free to fork the repo, open issues, and submit pull requests for
new pipeline blocks, optimizations, or fixes.

Before opening a PR, please format any C++/CUDA changes with the `.clang-format` config provided
at the project root (4-space indentation, no tabs):

```bash
clang-format -i <changed files>
```

and make sure the project still builds with `cmake --build build`.

## Copyright and license

HyperISP is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

For the complete license text, see the [LICENSE](LICENSE) file in the repository.
