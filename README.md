# PolUVR 🎶

[![PyPI version](https://badge.fury.io/py/PolUVR.svg?icon=si%3Apython)](https://badge.fury.io/py/PolUVR)
[![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Politrees/PolUVR)

## Overview

PolUVR is a Python-based audio separation tool that leverages advanced machine learning models to separate audio tracks into different stems, such as vocals, instrumental, drums, bass, and more. This project is a fork of the [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) repository, and it aims to provide a user-friendly interface for audio separation tasks.

---

## Installation 🛠️

### Hardware Acceleration Options

#### Nvidia GPU with CUDA

**Supported CUDA Versions:** 11.8 and 12.2

To verify successful configuration, run `PolUVR --env_info`. You should see the following log message:
```
ONNXruntime has CUDAExecutionProvider available, enabling acceleration
```

**Installation:**
```sh
pip install "PolUVR[gpu]"
```

#### Apple Silicon, macOS Sonoma+ with M1 or newer CPU (CoreML acceleration)

To verify successful configuration, run `PolUVR --env_info`. You should see the following log message:
```
ONNXruntime has CoreMLExecutionProvider available, enabling acceleration
```

**Installation:**
```sh
pip install "PolUVR[cpu]"
```

#### CPU-Only (No Hardware Acceleration)

**Installation:**
```sh
pip install "PolUVR[cpu]"
```

---

### FFmpeg Dependency

To check if `PolUVR` is correctly configured to use FFmpeg, run `PolUVR --env_info`. The log should show:
```
FFmpeg installed
```

If it says that FFmpeg is missing or an error occurs, install FFmpeg using the following commands:

**Debian/Ubuntu:**
* ```sh
  apt-get update; apt-get install -y ffmpeg
  ```
**macOS:**
* ```sh
  brew update; brew install ffmpeg
  ```
**Windows:**
* Follow this guide: [Install-FFmpeg-on-Windows](https://www.wikihow.com/Install-FFmpeg-on-Windows)

If you cloned the repository, you can use the following command to install FFmpeg:
```sh
PolUVR-ffmpeg
```

---

## GPU / CUDA Specific Installation Steps

In theory, installing `PolUVR` with the `[gpu]` extra should suffice. However, sometimes PyTorch and ONNX Runtime with CUDA support can be tricky. You may need to reinstall these packages directly:

```sh
pip uninstall torch onnxruntime
pip cache purge
pip install --force-reinstall torch torchvision torchaudio
pip install --force-reinstall onnxruntime-gpu
```

For the latest PyTorch version, use the command recommended by the [PyTorch installation wizard](https://pytorch.org/get-started/locally/).

### Multiple CUDA Library Versions

Depending on your environment, you may need specific CUDA library versions. For example, Google Colab uses CUDA 12 by default, but ONNX Runtime may still require CUDA 11 libraries. Install CUDA 11 libraries alongside CUDA 12:

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

```sh
usage: PolUVR-app [--share] [--open]

Params:
  --share                  Opens public access to the interface (for servers, Google Colab, Kaggle, etc.).
  --open                   Automatically opens the interface in a new browser tab.

```
Once the following output message `Running on local URL:  http://127.0.0.1:7860` or `Running on public URL: https://28425b3eb261b9ddc6.gradio.live` appears, you can click on the link to open a tab with the WebUI.

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
git clone https://github.com/YOUR_USERNAME/PolUVR.git
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

Install extra dependencies:
```sh
poetry install --extras "cpu"
```
or
```sh
poetry install --extras "gpu"
```

### Running the CLI Locally

```sh
PolUVR path/to/your/audio-file.wav
```

### Deactivate the Virtual Environment

```sh
conda deactivate
```

---

## Contributing 🤝

Contributions are welcome! Fork the repository, make your changes, and submit a pull request.

---

# Original Repository

This project is a fork of the original [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) repository.
