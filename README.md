# PolUVR 🎶

[![PyPI version](https://badge.fury.io/py/PolUVR.svg?icon=si%3Apython)](https://badge.fury.io/py/PolUVR)
[![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Politrees/PolUVR)

## Overview

**PolUVR** is a Python-based audio separation tool that leverages advanced machine learning models to separate audio tracks into distinct stems, such as vocals, instrumental, drums, bass, and more. Built as a fork of the [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), PolUVR offers enhanced usability, hardware acceleration, and a user-friendly Gradio interface.

---

## Key Features

- **Audio Separation:** Extract vocals, instrumental, drums, bass, and other stems.
- **Hardware Acceleration:** Supports CUDA (Nvidia GPUs) and CoreML (Apple Silicon).
- **Cross-Platform:** Works on Linux, macOS, and Windows.
- **Gradio Interface:** Easy-to-use web interface for audio separation.

---

## Installation 🛠️

### Hardware Acceleration Options

PolUVR supports multiple hardware acceleration options for optimal performance. To verify successful configuration, run:
```sh
PolUVR --env_info
```

| **Command**                 | **Expected Log Message**                                                   |
|-----------------------------|----------------------------------------------------------------------------|
| `pip install "PolUVR[gpu]"` | `ONNXruntime has CUDAExecutionProvider available, enabling acceleration`   |
| `pip install "PolUVR[cpu]"` | `ONNXruntime has CoreMLExecutionProvider available, enabling acceleration` |
| `pip install "PolUVR[cpu]"` | No hardware acceleration enabled                                           |

---

### FFmpeg Dependency

PolUVR relies on FFmpeg for audio processing. To check if FFmpeg is installed, run:
```sh
PolUVR --env_info
```
The log should show: `FFmpeg installed`

If FFmpeg is missing, install it using the following commands:

| **OS**            | **Command**                                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------|
| **Debian/Ubuntu** | `apt-get update; apt-get install -y ffmpeg`                                                       |
| **macOS**         | `brew update; brew install ffmpeg`                                                                |
| **Windows**       | Follow this guide: [Install FFmpeg on Windows](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) |

If you cloned the repository, you can install FFmpeg with:
```sh
PolUVR-ffmpeg
```

---

## GPU / CUDA Specific Installation Steps

While installing `PolUVR` with the `[gpu]` extra should suffice, sometimes PyTorch and ONNX Runtime with CUDA support require manual intervention. If you encounter issues, follow these steps:

```sh
pip uninstall torch onnxruntime
pip cache purge
pip install --force-reinstall torch torchvision torchaudio
pip install --force-reinstall onnxruntime-gpu
```

For the latest PyTorch version, use the command recommended by the [PyTorch installation wizard](https://pytorch.org/get-started/locally/).

### Multiple CUDA Library Versions

If you need to install multiple CUDA versions (e.g., CUDA 11 alongside CUDA 12), use:
```sh
apt update; apt install nvidia-cuda-toolkit
```

If you encounter errors like `Failed to load library` or `cannot open shared object file`, resolve them by running:
```sh
python -m pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
```

---

## Usage 🚀

### Gradio Interface

To launch the Gradio interface, use:
```sh
PolUVR-app [--share] [--open]
```

| **Parameter** | **Description**                                                                |
|---------------|--------------------------------------------------------------------------------|
| `--share`     | Opens public access to the interface (useful for servers, Google Colab, etc.). |
| `--open`      | Automatically opens the interface in a new browser tab.                        |

As soon as one of the following messages appears:
```
Running on local URL:  http://127.0.0.1:7860
```
```
Running on public URL: https://28425b3eb261b9ddc6.gradio.live
```
you can click on the link to open the WebUI.

---

## Requirements 📋

- Python >= 3.10
- Libraries: torch, onnx, onnxruntime, numpy, librosa, requests, six, tqdm, pydub

---

## Developing Locally

### Prerequisites

- Python 3.10 or newer
- Conda (recommended: [Miniforge](https://github.com/conda-forge/miniforge))

### Clone the Repository

```sh
git clone https://github.com/Bebra777228/PolUVR.git
cd PolUVR
```

### Create and Activate the Conda Environment

```sh
conda env create
conda activate PolUVR-dev
```

### Install Dependencies

```sh
poetry install
```

For extra dependencies, use:
```sh
poetry install --extras "cpu"
```
or
```sh
poetry install --extras "gpu"
```

### Running the Gradio interface Locally

```sh
PolUVR-app --open
```

### Deactivate the Virtual Environment

```sh
conda deactivate
```

---

## Contributing 🤝

Contributions are welcome! Fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to add.

---

## Acknowledgments

This project is a fork of the original [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) repository. Special thanks to the contributors of the original project for their foundational work.
