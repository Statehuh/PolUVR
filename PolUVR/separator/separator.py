from importlib import metadata, resources
import os
import sys
import platform
import subprocess
import time
import logging
import warnings
import importlib
import io
from typing import Optional
import hashlib
import json
import yaml
import requests
import torch
import torch.amp.autocast_mode as autocast_mode
import onnxruntime as ort
from tqdm import tqdm


class Separator:
    def __init__(
        self,
        log_level=logging.INFO,
        log_formatter=None,
        model_file_dir="/tmp/PolUVR-models/",
        output_dir=None,
        output_format="WAV",
        output_bitrate=None,
        normalization_threshold=0.9,
        amplification_threshold=0.0,
        output_single_stem=None,
        invert_using_spec=False,
        sample_rate=44100,
        use_soundfile=False,
        use_autocast=False,
        mdx_params={
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.25,
            "batch_size": 1,
            "enable_denoise": False,
        },
        vr_params={
            "batch_size": 1,
            "window_size": 512,
            "aggression": 5,
            "enable_tta": False,
            "enable_post_process": False,
            "post_process_threshold": 0.2,
            "high_end_process": False,
        },
        demucs_params={
            "segment_size": "Default",
            "shifts": 2,
            "overlap": 0.25,
            "segments_enabled": True,
        },
        mdxc_params={
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
        },
        info_only=True,
        quiet: bool = True,
    ):
        """Initialize the separator (with optional quieter mode)."""
        self.quiet = bool(quiet)

        # If quiet, raise minimum log level to WARNING to suppress info/debug noise
        if self.quiet:
            log_level = max(log_level, logging.WARNING)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        # Ensure a single stream handler and apply formatting
        self.log_handler = logging.StreamHandler()
        if self.log_formatter is None:
            self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.log_handler.setFormatter(self.log_formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(self.log_handler)
        else:
            # Avoid adding duplicate handlers in repeated imports/instantiations
            # Replace existing stream handler formatter if present
            for h in self.logger.handlers:
                if isinstance(h, logging.StreamHandler):
                    h.setFormatter(self.log_formatter)

        # Filter out noisy warnings from PyTorch and other libs if quiet or not in DEBUG
        if self.quiet or log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")

        # Lower noise from requests/urllib3 and onnxruntime if quiet
        if self.quiet:
            try:
                logging.getLogger("requests").setLevel(logging.WARNING)
                logging.getLogger("urllib3").setLevel(logging.WARNING)
            except Exception:
                pass
            try:
                # Try to reduce ONNX Runtime internal logging if API available
                if hasattr(ort, "set_default_logger_severity"):
                    # Severity: 0 (VERBOSE) .. 4 (FATAL). Set to 3 to show only ERROR/FATAL.
                    ort.set_default_logger_severity(3)
            except Exception:
                # don't break if older onnxruntime doesn't expose that API
                pass

        # Skip initialization logs if info_only is True
        if not info_only and not self.quiet:
            # try to display package version only when not quiet
            try:
                package_version = self.get_package_distribution("PolUVR").version
            except Exception:
                package_version = "unknown"
            self.logger.info(f"Separator version {package_version} instantiating with output_dir: {output_dir}, output_format: {output_format}")

        if output_dir is None:
            output_dir = os.getcwd()
            if not info_only and not self.quiet:
                self.logger.info("Output directory not specified. Using current working directory.")

        self.output_dir = output_dir

        # Check for environment variable to override model_file_dir
        env_model_dir = os.environ.get("POLUVR_MODEL_DIR")
        if env_model_dir:
            self.model_file_dir = env_model_dir
            if not self.quiet:
                self.logger.info(f"Using model directory from POLUVR_MODEL_DIR env var: {self.model_file_dir}")
            if not os.path.exists(self.model_file_dir):
                raise FileNotFoundError(f"The specified model directory does not exist: {self.model_file_dir}")
        else:
            if not self.quiet:
                self.logger.info(f"Using model directory from model_file_dir parameter: {model_file_dir}")
            self.model_file_dir = model_file_dir

        # Create the model directory and output directory if they do not exist
        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_format = output_format
        self.output_bitrate = output_bitrate

        if self.output_format is None:
            self.output_format = "WAV"

        self.normalization_threshold = normalization_threshold
        if normalization_threshold <= 0 or normalization_threshold > 1:
            raise ValueError("The normalization_threshold must be greater than 0 and less than or equal to 1.")

        self.amplification_threshold = amplification_threshold
        if amplification_threshold < 0 or amplification_threshold > 1:
            raise ValueError("The amplification_threshold must be greater than or equal to 0 and less than or equal to 1.")

        self.output_single_stem = output_single_stem
        if output_single_stem is not None and not self.quiet:
            self.logger.debug(f"Single stem output requested, so only one output file ({output_single_stem}) will be written")

        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec and not self.quiet:
            self.logger.debug("Secondary step will be inverted using spectrogram rather than waveform. This may improve quality but is slightly slower.")

        try:
            self.sample_rate = int(sample_rate)
            if self.sample_rate <= 0:
                raise ValueError(f"The sample rate setting is {self.sample_rate} but it must be a non-zero whole number.")
            if self.sample_rate > 12800000:
                raise ValueError(f"The sample rate setting is {self.sample_rate}. Enter something less ambitious.")
        except ValueError:
            raise ValueError("The sample rate must be a non-zero whole number. Please provide a valid integer.")

        self.use_soundfile = use_soundfile
        self.use_autocast = use_autocast

        # These are parameters which users may want to configure so we expose them to the top-level Separator class,
        # even though they are specific to a single model architecture
        self.arch_specific_params = {"MDX": mdx_params, "VR": vr_params, "Demucs": demucs_params, "MDXC": mdxc_params}

        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None

        self.onnx_execution_provider = None
        self.model_instance = None

        self.model_is_uvr_vip = False
        self.model_friendly_name = None

        # Whether to disable tqdm progress bars
        self._disable_tqdm = bool(self.quiet)

        if not info_only:
            self.setup_accelerated_inferencing_device()

    def setup_accelerated_inferencing_device(self):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
        """
        system_info = self.get_system_info()
        self.check_ffmpeg_installed()
        self.log_onnxruntime_packages()
        self.setup_torch_device(system_info)

    def get_system_info(self):
        """
        This method logs the system information, including the operating system, CPU architecture and Python version
        """
        os_name = platform.system()
        os_version = platform.version()
        if not self.quiet:
            self.logger.info(f"Operating System: {os_name} {os_version}")

        system_info = platform.uname()
        if not self.quiet:
            self.logger.info(f"System: {system_info.system} Node: {system_info.node} Release: {system_info.release} Machine: {system_info.machine} Proc: {system_info.processor}")

        python_version = platform.python_version()
        if not self.quiet:
            self.logger.info(f"Python Version: {python_version}")

        pytorch_version = torch.__version__
        if not self.quiet:
            self.logger.info(f"PyTorch Version: {pytorch_version}")
        return system_info

    def check_ffmpeg_installed(self):
        """
        This method checks if ffmpeg is installed and logs its version.
        """
        try:
            ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
            first_line = ffmpeg_version_output.splitlines()[0]
            if not self.quiet:
                self.logger.info(f"FFmpeg installed: {first_line}")
        except FileNotFoundError:
            self.logger.error("FFmpeg is not installed. Please install FFmpeg to use this package.")
            # Raise an exception if this is being run by a user, as ffmpeg is required for pydub to write audio
            # but if we're just running unit tests in CI, no reason to throw
            if "PYTEST_CURRENT_TEST" not in os.environ:
                raise

    def log_onnxruntime_packages(self):
        """
        This method logs the ONNX Runtime package versions, including the GPU and Silicon packages if available.
        """
        onnxruntime_gpu_package = self.get_package_distribution("onnxruntime-gpu")
        onnxruntime_silicon_package = self.get_package_distribution("onnxruntime-silicon")
        onnxruntime_cpu_package = self.get_package_distribution("onnxruntime")

        if not self.quiet and onnxruntime_gpu_package is not None:
            self.logger.info(f"ONNX Runtime GPU package installed with version: {onnxruntime_gpu_package.version}")
        if not self.quiet and onnxruntime_silicon_package is not None:
            self.logger.info(f"ONNX Runtime Silicon package installed with version: {onnxruntime_silicon_package.version}")
        if not self.quiet and onnxruntime_cpu_package is not None:
            self.logger.info(f"ONNX Runtime CPU package installed with version: {onnxruntime_cpu_package.version}")

    def setup_torch_device(self, system_info):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
        """
        hardware_acceleration_enabled = False
        try:
            ort_providers = ort.get_available_providers()
        except Exception:
            ort_providers = []

        self.torch_device_cpu = torch.device("cpu")

        if torch.cuda.is_available():
            self.configure_cuda(ort_providers)
            hardware_acceleration_enabled = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and system_info.processor == "arm":
            self.configure_mps(ort_providers)
            hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled and not self.quiet:
            self.logger.info("No hardware acceleration could be configured, running in CPU mode")
        if not hardware_acceleration_enabled:
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    def configure_cuda(self, ort_providers):
        """
        This method configures the CUDA device for PyTorch and ONNX Runtime, if available.
        """
        if not self.quiet:
            self.logger.info("CUDA is available in Torch, setting Torch device to CUDA")
        self.torch_device = torch.device("cuda")
        if "CUDAExecutionProvider" in ort_providers:
            if not self.quiet:
                self.logger.info("ONNXruntime has CUDAExecutionProvider available, enabling acceleration")
            self.onnx_execution_provider = ["CUDAExecutionProvider"]
        else:
            self.logger.warning("CUDAExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")

    def configure_mps(self, ort_providers):
        """
        This method configures the Apple Silicon MPS/CoreML device for PyTorch and ONNX Runtime, if available.
        """
        if not self.quiet:
            self.logger.info("Apple Silicon MPS/CoreML is available in Torch and processor is ARM, setting Torch device to MPS")
        self.torch_device_mps = torch.device("mps")
        self.torch_device = self.torch_device_mps
        if "CoreMLExecutionProvider" in ort_providers:
            if not self.quiet:
                self.logger.info("ONNXruntime has CoreMLExecutionProvider available, enabling acceleration")
            self.onnx_execution_provider = ["CoreMLExecutionProvider"]
        else:
            self.logger.warning("CoreMLExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled")

    def get_package_distribution(self, package_name):
        """
        This method returns the package distribution for a given package name if installed, or None otherwise.
        """
        try:
            return metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            if not self.quiet:
                self.logger.debug(f"Python package: {package_name} not installed")
            return None

    def get_model_hash(self, model_path):
        """
        This method returns the MD5 hash of a given model file.
        """
        if not self.quiet:
            self.logger.debug(f"Calculating hash of model file {model_path}")

        # Use the specific byte count from the original logic
        BYTES_TO_HASH = 10000 * 1024  # 10,240,000 bytes
        try:
            file_size = os.path.getsize(model_path)
            with open(model_path, "rb") as f:
                if file_size < BYTES_TO_HASH:
                    if not self.quiet:
                        self.logger.debug(f"File size {file_size} < {BYTES_TO_HASH}, hashing entire file.")
                    hash_value = hashlib.md5(f.read()).hexdigest()
                else:
                    seek_pos = file_size - BYTES_TO_HASH
                    if not self.quiet:
                        self.logger.debug(f"File size {file_size} >= {BYTES_TO_HASH}, seeking to {seek_pos} and hashing remaining bytes.")
                    f.seek(seek_pos, io.SEEK_SET)
                    hash_value = hashlib.md5(f.read()).hexdigest()

            if not self.quiet:
                self.logger.info(f"Hash of model file {model_path} is {hash_value}")
            return hash_value

        except FileNotFoundError:
            self.logger.error(f"Model file not found at {model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error calculating hash for {model_path}: {e}")
            raise

    def download_file_if_not_exists(self, url, output_path):
        """
        This method downloads a file from a given URL to a given output path, if the file does not already exist.
        Progress bars are disabled when quiet.
        """

        if os.path.isfile(output_path):
            if not self.quiet:
                self.logger.debug(f"File already exists at {output_path}, skipping download")
            return

        if not self.quiet:
            self.logger.debug(f"Downloading file from {url} to {output_path} with timeout 300s")
        try:
            response = requests.get(url, stream=True, timeout=300)
        except Exception as e:
            raise RuntimeError(f"Failed to request download from {url}: {e}")

        if response.status_code == 200:
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            progress_message = "Model Download: " + os.path.basename(output_path)
            # Respect quiet mode by disabling tqdm when requested
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=progress_message, disable=self._disable_tqdm)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        if not self._disable_tqdm:
                            progress_bar.update(len(chunk))
                        f.write(chunk)
            if not self._disable_tqdm:
                progress_bar.close()
        else:
            raise RuntimeError(f"Failed to download file from {url}, response code: {response.status_code}")

    def list_supported_model_files(self):
        """
        This method lists the supported model files for PolUVR, by fetching the same file UVR uses to list these.
        Also includes model performance scores where available.
        """
        download_checks_path = os.path.join(self.model_file_dir, "model_list_links.json")
        self.download_file_if_not_exists("https://raw.githubusercontent.com/Politrees/UVR_resources/main/UVR_resources/model_list_links.json", download_checks_path)

        model_downloads_list = json.load(open(download_checks_path, encoding="utf-8"))
        if not self.quiet:
            self.logger.debug("UVR model download list loaded")

        # Load the model scores with error handling
        model_scores = {}
        try:
            with resources.open_text("PolUVR", "models-scores.json") as f:
                model_scores = json.load(f)
            if not self.quiet:
                self.logger.debug("Model scores loaded")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to load model scores: {str(e)}")
            if not self.quiet:
                self.logger.warning("Continuing without model scores")
        except Exception:
            # If package resources or file not present, continue silently if quiet
            if not self.quiet:
                self.logger.debug("No local model score resource available; continuing without it")

        # Only show Demucs v4 models as we've only implemented support for v4
        filtered_demucs_v4 = {key: value for key, value in model_downloads_list.get("demucs_download_list", {}).items() if key.startswith("Demucs v4")}

        # Modified Demucs handling to use YAML files as identifiers and include download files
        demucs_models = {}
        for name, files in filtered_demucs_v4.items():
            yaml_file = next((filename for filename in files.keys() if filename.endswith(".yaml")), None)
            if yaml_file:
                model_score_data = model_scores.get(yaml_file, {})
                demucs_models[name] = {
                    "filename": yaml_file,
                    "scores": model_score_data.get("median_scores", {}),
                    "stems": model_score_data.get("stems", []),
                    "target_stem": model_score_data.get("target_stem"),
                    "download_files": list(files.values()),
                }
        if not self.quiet:
            self.logger.debug("PolUVR model list loaded")

        model_files_grouped_by_type = {
            "VR": {
                name: {
                    "filename": next(iter(files.keys())),
                    "scores": model_scores.get(next(iter(files.keys())), {}).get("median_scores", {}),
                    "stems": model_scores.get(next(iter(files.keys())), {}).get("stems", []),
                    "target_stem": model_scores.get(next(iter(files.keys())), {}).get("target_stem"),
                    "download_files": list(files.values()),
                }
                for name, files in model_downloads_list.get("vr_download_list", {}).items()
            },
            "MDX": {
                name: {
                    "filename": next(iter(files.keys())),
                    "scores": model_scores.get(next(iter(files.keys())), {}).get("median_scores", {}),
                    "stems": model_scores.get(next(iter(files.keys())), {}).get("stems", []),
                    "target_stem": model_scores.get(next(iter(files.keys())), {}).get("target_stem"),
                    "download_files": list(files.values()),
                }
                for name, files in {
                    **model_downloads_list.get("mdx_download_list", {}),
                    **model_downloads_list.get("mdx_download_vip_list", {}),
                }.items()
            },
            "Demucs": demucs_models,
            "MDXC": {
                name: {
                    "filename": next(iter(files.keys())),
                    "scores": model_scores.get(next(iter(files.keys())), {}).get("median_scores", {}),
                    "stems": model_scores.get(next(iter(files.keys())), {}).get("stems", []),
                    "target_stem": model_scores.get(next(iter(files.keys())), {}).get("target_stem"),
                    "download_files": list(files.values()),
                }
                for name, files in {
                    **model_downloads_list.get("mdx23c_download_list", {}),
                    **model_downloads_list.get("mdx23c_download_vip_list", {}),
                    **model_downloads_list.get("roformer_download_list", {}),
                }.items()
            },
        }

        return model_files_grouped_by_type

    def print_uvr_vip_message(self):
        """
        This method prints a message to the user if they have downloaded a VIP model, reminding them to support Anjok07 on Patreon.
        Shown as a warning so it is visible even in quiet mode.
        """
        if self.model_is_uvr_vip:
            # Keep it as a warning to ensure it surfaces in quiet mode
            self.logger.warning(f"The model: '{self.model_friendly_name}' is a VIP model, intended by Anjok07 for access by paying subscribers only.")
            self.logger.warning("If you are not already subscribed, please consider supporting the developer of UVR, Anjok07 by subscribing here: https://patreon.com/uvr")

    def download_model_files(self, model_filename):
        """
        This method downloads the model files for a given model filename, if they are not already present.
        Returns tuple of (model_filename, model_type, model_friendly_name, model_path, yaml_config_filename)
        """
        model_path = os.path.join(self.model_file_dir, f"{model_filename}")

        supported_model_files_grouped = self.list_supported_model_files()
        yaml_config_filename = None

        if not self.quiet:
            self.logger.debug(f"Searching for model_filename {model_filename} in supported_model_files_grouped")

        # Iterate through model types (VR, MDX, Demucs, MDXC)
        for model_type, models in supported_model_files_grouped.items():
            for model_friendly_name, model_info in models.items():
                if model_filename == model_info["filename"]:
                    if not self.quiet:
                        self.logger.debug(f"Found matching model: {model_friendly_name}")
                    self.model_friendly_name = model_friendly_name

                    self.model_is_uvr_vip = "VIP" in model_friendly_name
                    self.print_uvr_vip_message()

                    for url in model_info["download_files"]:
                        filename = url.split("/")[-1]
                        download_path = os.path.join(self.model_file_dir, filename)

                        if model_type == "MDXC" and url.endswith(".yaml"):
                            yaml_config_filename = filename

                        self.download_file_if_not_exists(url, download_path)

                    return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename

        raise ValueError(f"Model file {model_filename} not found in supported model files")

    def load_model_data_from_yaml(self, yaml_config_filename):
        """
        This method loads model-specific parameters from the YAML file for that model.
        """
        if not os.path.exists(yaml_config_filename):
            model_data_yaml_filepath = os.path.join(self.model_file_dir, yaml_config_filename)
        else:
            model_data_yaml_filepath = yaml_config_filename

        if not self.quiet:
            self.logger.debug(f"Loading model data from YAML at path {model_data_yaml_filepath}")

        model_data = yaml.load(open(model_data_yaml_filepath, encoding="utf-8"), Loader=yaml.FullLoader)
        if not self.quiet:
            self.logger.debug(f"Model data loaded from YAML file: {model_data}")

        if "roformer" in model_data_yaml_filepath:
            model_data["is_roformer"] = True

        return model_data

    def load_model_data_using_hash(self, model_path):
        """
        This method loads model-specific parameters from UVR model data files by calculating the model hash.
        """
        model_data_url_prefix = "https://raw.githubusercontent.com/Politrees/UVR_resources/main/UVR_resources/model_data"

        vr_model_data_url = f"{model_data_url_prefix}/vr_model_data.json"
        mdx_model_data_url = f"{model_data_url_prefix}/mdx_model_data.json"

        if not self.quiet:
            self.logger.debug("Calculating MD5 hash for model file to identify model parameters from UVR data...")
        model_hash = self.get_model_hash(model_path)
        if not self.quiet:
            self.logger.debug(f"Model {model_path} has hash {model_hash}")

        vr_model_data_path = os.path.join(self.model_file_dir, "vr_model_data.json")
        if not self.quiet:
            self.logger.debug(f"VR model data path set to {vr_model_data_path}")
        self.download_file_if_not_exists(vr_model_data_url, vr_model_data_path)

        mdx_model_data_path = os.path.join(self.model_file_dir, "mdx_model_data.json")
        if not self.quiet:
            self.logger.debug(f"MDX model data path set to {mdx_model_data_path}")
        self.download_file_if_not_exists(mdx_model_data_url, mdx_model_data_path)

        if not self.quiet:
            self.logger.debug("Loading MDX and VR model parameters from UVR model data files...")
        vr_model_data_object = json.load(open(vr_model_data_path, encoding="utf-8"))
        mdx_model_data_object = json.load(open(mdx_model_data_path, encoding="utf-8"))

        if model_hash in mdx_model_data_object:
            model_data = mdx_model_data_object[model_hash]
        elif model_hash in vr_model_data_object:
            model_data = vr_model_data_object[model_hash]
        else:
            raise ValueError(f"Unsupported Model File: parameters for MD5 hash {model_hash} could not be found in UVR model data file for MDX or VR arch.")

        if not self.quiet:
            self.logger.debug(f"Model data loaded using hash {model_hash}: {model_data}")

        return model_data

    def load_model(self, model_filename="model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"):
        """
        Instantiate the architecture-specific separation class, downloading model files first if needed.
        """
        if not self.quiet:
            self.logger.info(f"Loading model {model_filename}...")

        load_model_start_time = time.perf_counter()

        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)
        model_name = model_filename.split(".")[0]
        if not self.quiet:
            self.logger.debug(f"Model downloaded, friendly name: {model_friendly_name}, model_path: {model_path}")

        if model_path.lower().endswith(".yaml"):
            yaml_config_filename = model_path

        if yaml_config_filename is not None:
            model_data = self.load_model_data_from_yaml(yaml_config_filename)
        else:
            model_data = self.load_model_data_using_hash(model_path)

        common_params = {
            "logger": self.logger,
            "log_level": self.log_level,
            "torch_device": self.torch_device,
            "torch_device_cpu": self.torch_device_cpu,
            "torch_device_mps": self.torch_device_mps,
            "onnx_execution_provider": self.onnx_execution_provider,
            "model_name": model_name,
            "model_path": model_path,
            "model_data": model_data,
            "output_format": self.output_format,
            "output_bitrate": self.output_bitrate,
            "output_dir": self.output_dir,
            "normalization_threshold": self.normalization_threshold,
            "amplification_threshold": self.amplification_threshold,
            "output_single_stem": self.output_single_stem,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
            "use_soundfile": self.use_soundfile,
        }

        separator_classes = {
            "MDX": "mdx_separator.MDXSeparator",
            "VR": "vr_separator.VRSeparator",
            "Demucs": "demucs_separator.DemucsSeparator",
            "MDXC": "mdxc_separator.MDXCSeparator",
        }

        if model_type not in self.arch_specific_params or model_type not in separator_classes:
            raise ValueError(f"Model type not supported (yet): {model_type}")

        if model_type == "Demucs" and sys.version_info < (3, 10):
            raise Exception("Demucs models require Python version 3.10 or newer.")

        if not self.quiet:
            self.logger.debug(f"Importing module for model type {model_type}: {separator_classes[model_type]}")

        module_name, class_name = separator_classes[model_type].split(".")
        module = importlib.import_module(f"PolUVR.separator.architectures.{module_name}")
        separator_class = getattr(module, class_name)

        if not self.quiet:
            self.logger.debug(f"Instantiating separator class for model type {model_type}: {separator_class}")
        self.model_instance = separator_class(common_config=common_params, arch_config=self.arch_specific_params[model_type])

        if not self.quiet:
            self.logger.debug("Loading model completed.")
            self.logger.info(f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - load_model_start_time)))}')

    def separate(self, audio_file_path, custom_output_names=None):
        """
        Separates the audio file(s) into different stems using the loaded model.
        """
        if not (self.torch_device and self.model_instance):
            raise ValueError("Initialization failed or model not loaded. Please load a model before attempting to separate.")

        if isinstance(audio_file_path, str):
            audio_file_path = [audio_file_path]

        output_files = []

        for path in audio_file_path:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith((".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3")):
                            full_path = os.path.join(root, file)
                            if not self.quiet:
                                self.logger.info(f"Processing file: {full_path}")
                            try:
                                files_output = self._separate_file(full_path, custom_output_names)
                                output_files.extend(files_output)
                            except Exception as e:
                                self.logger.error(f"Failed to process file {full_path}: {e}")
            else:
                if not self.quiet:
                    self.logger.info(f"Processing file: {path}")
                try:
                    files_output = self._separate_file(path, custom_output_names)
                    output_files.extend(files_output)
                except Exception as e:
                    self.logger.error(f"Failed to process file {path}: {e}")

        return output_files

    def _separate_file(self, audio_file_path, custom_output_names=None):
        """
        Internal method to handle separation for a single audio file.
        """
        if not self.quiet:
            self.logger.info(f"Starting separation process for audio_file_path: {audio_file_path}")
        separate_start_time = time.perf_counter()

        if not self.quiet:
            self.logger.debug(f"Normalization threshold set to {self.normalization_threshold}, waveform will be lowered to this max amplitude to avoid clipping.")
            self.logger.debug(f"Amplification threshold set to {self.amplification_threshold}, waveform will be scaled up to this max amplitude if below it.")

        output_files = None
        try:
            if self.use_autocast and autocast_mode.is_autocast_available(self.torch_device.type):
                if not self.quiet:
                    self.logger.debug("Autocast available.")
                with autocast_mode.autocast(self.torch_device.type):
                    output_files = self.model_instance.separate(audio_file_path, custom_output_names)
            else:
                if not self.quiet:
                    self.logger.debug("Autocast unavailable.")
                output_files = self.model_instance.separate(audio_file_path, custom_output_names)
        finally:
            # Always try to clear caches and file-specific state
            try:
                self.model_instance.clear_gpu_cache()
            except Exception:
                pass
            try:
                self.model_instance.clear_file_specific_paths()
            except Exception:
                pass

        self.print_uvr_vip_message()

        if not self.quiet:
            self.logger.debug("Separation process completed.")
            self.logger.info(f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - separate_start_time)))}')

        return output_files

    def download_model_and_data(self, model_filename):
        """
        Downloads the model file without loading it into memory.
        """
        if not self.quiet:
            self.logger.info(f"Downloading model {model_filename}...")

        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)

        if model_path.lower().endswith(".yaml"):
            yaml_config_filename = model_path

        if yaml_config_filename is not None:
            model_data = self.load_model_data_from_yaml(yaml_config_filename)
        else:
            model_data = self.load_model_data_using_hash(model_path)

        model_data_dict_size = len(model_data)

        if not self.quiet:
            self.logger.info(f"Model downloaded, type: {model_type}, friendly name: {model_friendly_name}, model_path: {model_path}, model_data: {model_data_dict_size} items")

    def get_simplified_model_list(self, filter_sort_by: Optional[str] = None):
        """
        Returns a simplified, user-friendly list of models with their key metrics.
        Optionally sorts the list based on the specified criteria.
        """
        model_files = self.list_supported_model_files()
        simplified_list = {}

        for model_type, models in model_files.items():
            for name, data in models.items():
                filename = data["filename"]
                scores = data.get("scores") or {}
                stems = data.get("stems") or []
                target_stem = data.get("target_stem")

                stems_with_scores = []
                stem_sdr_dict = {}

                for stem in stems:
                    stem_scores = scores.get(stem, {})
                    stem_display = f"{stem}*" if stem == target_stem else stem

                    if isinstance(stem_scores, dict) and "SDR" in stem_scores:
                        sdr = round(stem_scores["SDR"], 1)
                        stems_with_scores.append(f"{stem_display} ({sdr})")
                        stem_sdr_dict[stem.lower()] = sdr
                    else:
                        stems_with_scores.append(stem_display)
                        stem_sdr_dict[stem.lower()] = None

                if not stems_with_scores:
                    stems_with_scores = ["Unknown"]
                    stem_sdr_dict["unknown"] = None

                simplified_list[filename] = {"Name": name, "Type": model_type, "Stems": stems_with_scores, "SDR": stem_sdr_dict}

        if filter_sort_by:
            if filter_sort_by == "name":
                return dict(sorted(simplified_list.items(), key=lambda x: x[1]["Name"]))
            elif filter_sort_by == "filename":
                return dict(sorted(simplified_list.items()))
            else:
                sort_by_lower = filter_sort_by.lower()
                filtered_list = {k: v for k, v in simplified_list.items() if sort_by_lower in v["SDR"]}

                def sort_key(item):
                    sdr = item[1]["SDR"][sort_by_lower]
                    return (0 if sdr is None else 1, sdr if sdr is not None else float("-inf"))

                return dict(sorted(filtered_list.items(), key=sort_key, reverse=True))

        return simplified_list
