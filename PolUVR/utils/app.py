import os
import re
import sys
import torch
import shutil
import logging
import subprocess
import gradio as gr

from PolUVR.separator import Separator
from UVR_resources import DEMUCS_v4_MODELS, VR_ARCH_MODELS, MDXNET_MODELS, MDX23C_MODELS, ROFORMER_MODELS

device = "cuda" if torch.cuda.is_available() else "cpu"
use_autocast = device == "cuda"

OUTPUT_FORMAT = ["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "ac3"]

def print_message(input_file, model_name):
    """Prints information about the audio separation process."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print("\n")
    print("🎵 PolUVR 🎵")
    print("Input file:", base_name)
    print("Model used:", model_name)
    print("Audio separation in progress...")

def prepare_output_dir(input_file, output_dir):
    """Creates a directory to save the results and clears it if it already exists."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = os.path.join(output_dir, base_name)
    try:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    except Exception as e:
        raise RuntimeError(f"Error creating output directory {out_dir}: {e}") from e
    return out_dir

def rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model):
    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {
        "Vocals": vocals_stem.replace("NAME", base_name).replace("STEM", "Vocals").replace("MODEL", model),
        "Instrumental": instrumental_stem.replace("NAME", base_name).replace("STEM", "Instrumental").replace("MODEL", model),
        "Drums": drums_stem.replace("NAME", base_name).replace("STEM", "Drums").replace("MODEL", model),
        "Bass": bass_stem.replace("NAME", base_name).replace("STEM", "Bass").replace("MODEL", model),
        "Other": other_stem.replace("NAME", base_name).replace("STEM", "Other").replace("MODEL", model),
        "Guitar": guitar_stem.replace("NAME", base_name).replace("STEM", "Guitar").replace("MODEL", model),
        "Piano": piano_stem.replace("NAME", base_name).replace("STEM", "Piano").replace("MODEL", model),
    }
    return stems

def leaderboard(list_filter, list_limit):
    try:
        result = subprocess.run(
            ["PolUVR", "-l", f"--list_filter={list_filter}", f"--list_limit={list_limit}"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return "<table border='1'>" + "".join(
            f"<tr style='{'font-weight: bold; font-size: 1.2em;' if i == 0 else ''}'>" +
            "".join(f"<td>{cell}</td>" for cell in re.split(r"\s{2,}", line.strip())) +
            "</tr>"
            for i, line in enumerate(re.findall(r"^(?!-+)(.+)$", result.stdout.strip(), re.MULTILINE))
        ) + "</table>"

    except Exception as e:
        return f"Error: {e}"

def roformer_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the Roformer model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
    print_message(audio, model_key)
    model = ROFORMER_MODELS[model_key]
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdxc_params={
                "segment_size": seg_size,
                "override_model_segment_size": override_seg_size,
                "batch_size": batch_size,
                "overlap": overlap,
                "pitch_shift": pitch_shift,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"Error separating audio with Roformer: {e}") from e

def mdx23c_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the MDX23C model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
    print_message(audio, model_key)
    model = MDX23C_MODELS[model_key]
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdxc_params={
                "segment_size": seg_size,
                "override_model_segment_size": override_seg_size,
                "batch_size": batch_size,
                "overlap": overlap,
                "pitch_shift": pitch_shift,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"Error separating audio with MDX23C: {e}") from e

def mdx_separator(audio, model_key, hop_length, seg_size, overlap, denoise, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the MDX-NET model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
    print_message(audio, model_key)
    model = MDXNET_MODELS[model_key]
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdx_params={
                "hop_length": hop_length,
                "segment_size": seg_size,
                "overlap": overlap,
                "batch_size": batch_size,
                "enable_denoise": denoise,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"Error separating audio with MDX-NET: {e}") from e

def vr_separator(audio, model_key, window_size, aggression, tta, post_process, post_process_threshold, high_end_process, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the VR ARCH model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
    print_message(audio, model_key)
    model = VR_ARCH_MODELS[model_key]
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            vr_params={
                "batch_size": batch_size,
                "window_size": window_size,
                "aggression": aggression,
                "enable_tta": tta,
                "enable_post_process": post_process,
                "post_process_threshold": post_process_threshold,
                "high_end_process": high_end_process,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"Error separating audio with VR ARCH: {e}") from e

def demucs_separator(audio, model_key, seg_size, shifts, overlap, segments_enabled, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the Demucs model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
    print_message(audio, model_key)
    model = DEMUCS_v4_MODELS[model_key]
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            demucs_params={
                "segment_size": seg_size,
                "shifts": shifts,
                "overlap": overlap,
                "segments_enabled": segments_enabled,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]

        if model_key == "htdemucs_6s":
            return stems[0], stems[1], stems[2], stems[3], stems[4], stems[5]
        return stems[0], stems[1], stems[2], stems[3], None, None
    except Exception as e:
        raise RuntimeError(f"Error separating audio with Demucs: {e}") from e

def update_stems(model):
    """Updates the visibility of output stems based on the selected Demucs model."""
    if model == "htdemucs_6s":
        return gr.update(visible=True)
    return gr.update(visible=False)

def show_hide_params(param):
    """Updates the visibility of a parameter based on the checkbox state."""
    return gr.update(visible=param)

def clear_models(model_dir):
    """Deletes all model files from the specified directory."""
    try:
        for filename in os.listdir(model_dir):
            if filename.endswith((".th", ".pth", ".onnx", ".ckpt", ".json", ".yaml")):
                file_path = os.path.join(model_dir, filename)
                os.remove(file_path)
        return "Models successfully cleared from memory."
    except Exception as e:
        return f"Error deleting models: {e}"

def PolUVR_UI(default_model_file_dir="/tmp/PolUVR-models/", default_output_dir="output"):
    with gr.Tab("Roformer"):
        with gr.Group():
            with gr.Row():
                roformer_model = gr.Dropdown(value="MelBand Roformer Kim | Big Beta v5e FT by Unwa", label="Model", choices=list(ROFORMER_MODELS.keys()), scale=3)
                roformer_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", info="Select the format for saving results.", scale=1)
            with gr.Accordion("Additional Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        roformer_override_seg_size = gr.Checkbox(value=False, label="Override Segment Size", info="Use a custom segment size instead of the default value.")
                        with gr.Row():
                            roformer_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Increasing the size can improve quality but requires more resources.", visible=False)
                            roformer_overlap = gr.Slider(minimum=2, maximum=10, step=1, value=8, label="Overlap", info="Decreasing overlap improves quality but slows down processing.")
                            roformer_pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Pitch Shift", info="Pitch shifting can improve separation for certain types of vocals.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            roformer_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Increasing batch size speeds up processing but requires more memory.")
                            roformer_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            roformer_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            roformer_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            roformer_button = gr.Button("Start Separation", variant="primary")
        with gr.Row():
            roformer_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            roformer_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("MDX23C"):
        with gr.Group():
            with gr.Row():
                mdx23c_model = gr.Dropdown(value="MDX23C InstVoc HQ", label="Model", choices=list(MDX23C_MODELS.keys()), scale=3)
                mdx23c_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", info="Select the format for saving results.", scale=1)
            with gr.Accordion("Additional Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        mdx23c_override_seg_size = gr.Checkbox(value=False, label="Override Segment Size", info="Use a custom segment size instead of the default value.")
                        with gr.Row():
                            mdx23c_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Increasing the size can improve quality but requires more resources.", visible=False)
                            mdx23c_overlap = gr.Slider(minimum=2, maximum=50, step=1, value=8, label="Overlap", info="Increasing overlap improves quality but slows down processing.")
                            mdx23c_pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Pitch Shift", info="Pitch shifting can improve separation for certain types of vocals.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            mdx23c_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Increasing batch size speeds up processing but requires more memory.")
                            mdx23c_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            mdx23c_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            mdx23c_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            mdx23c_button = gr.Button("Start Separation", variant="primary")
        with gr.Row():
            mdx23c_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            mdx23c_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("MDX-NET"):
        with gr.Group():
            with gr.Row():
                mdx_model = gr.Dropdown(value="UVR-MDX-NET Inst HQ 5", label="Model", choices=list(MDXNET_MODELS.keys()), scale=3)
                mdx_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", info="Select the format for saving results.", scale=1)
            with gr.Accordion("Additional Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        mdx_denoise = gr.Checkbox(value=False, label="Denoise", info="Enable denoising after separation.")
                        with gr.Row():
                            mdx_hop_length = gr.Slider(minimum=32, maximum=2048, step=32, value=1024, label="Hop Length", info="Parameter affecting separation accuracy.")
                            mdx_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Increasing the size can improve quality but requires more resources.")
                            mdx_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Overlap", info="Increasing overlap improves quality but slows down processing.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            mdx_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Increasing batch size speeds up processing but requires more memory.")
                            mdx_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            mdx_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            mdx_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            mdx_button = gr.Button("Start Separation", variant="primary")
        with gr.Row():
            mdx_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            mdx_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("VR ARCH"):
        with gr.Group():
            with gr.Row():
                vr_model = gr.Dropdown(value="1_HP-UVR", label="Model", choices=list(VR_ARCH_MODELS.keys()), scale=3)
                vr_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", info="Select the format for saving results.", scale=1)
            with gr.Accordion("Additional Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            vr_post_process = gr.Checkbox(value=False, label="Post-Process", info="Enable additional processing to improve separation quality.")
                            vr_tta = gr.Checkbox(value=False, label="TTA", info="Enable test-time augmentation for better quality.")
                            vr_high_end_process = gr.Checkbox(value=False, label="High-End Process", info="Restore missing high frequencies.")
                        with gr.Row():
                            vr_post_process_threshold = gr.Slider(minimum=0.1, maximum=0.3, step=0.1, value=0.2, label="Post-Process Threshold", info="Threshold for applying post-processing.", visible=False)
                            vr_window_size = gr.Slider(minimum=320, maximum=1024, step=32, value=512, label="Window Size", info="Decreasing window size improves quality but slows down processing.")
                            vr_aggression = gr.Slider(minimum=1, maximum=100, step=1, value=5, label="Aggression", info="Intensity of the main stem separation.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            vr_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Increasing batch size speeds up processing but requires more memory.")
                            vr_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            vr_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            vr_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            vr_button = gr.Button("Start Separation", variant="primary")
        with gr.Row():
            vr_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            vr_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("Demucs"):
        with gr.Group():
            with gr.Row():
                demucs_model = gr.Dropdown(value="htdemucs_ft", label="Model", choices=list(DEMUCS_v4_MODELS.keys()), scale=3)
                demucs_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", info="Select the format for saving results.", scale=1)
            with gr.Accordion("Additional Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        demucs_segments_enabled = gr.Checkbox(value=True, label="Segment Processing", info="Enable processing audio in segments.")
                        with gr.Row():
                            demucs_seg_size = gr.Slider(minimum=1, maximum=100, step=1, value=40, label="Segment Size", info="Increasing segment size improves quality but slows down processing.")
                            demucs_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Overlap", info="Increasing overlap improves quality but slows down processing.")
                            demucs_shifts = gr.Slider(minimum=0, maximum=20, step=1, value=2, label="Shifts", info="Increasing shifts improves quality but slows down processing.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            demucs_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            demucs_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            demucs_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            demucs_button = gr.Button("Start Separation", variant="primary")
        with gr.Row():
            demucs_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            demucs_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)
        with gr.Row():
            demucs_stem3 = gr.Audio(label="Stem 3", type="filepath", interactive=False)
            demucs_stem4 = gr.Audio(label="Stem 4", type="filepath", interactive=False)
        with gr.Row(visible=False) as stem6:
            demucs_stem5 = gr.Audio(label="Stem 5", type="filepath", interactive=False)
            demucs_stem6 = gr.Audio(label="Stem 6", type="filepath", interactive=False)

    with gr.Tab("Settings"):
        with gr.Row():
            with gr.Column(variant="panel"):
                model_file_dir = gr.Textbox(value=default_model_file_dir, label="Model Directory", info="Specify the path to store model files.", placeholder="models/UVR_models")
                gr.HTML("""<div style="margin: -10px 0!important; text-align: center">The button below will delete all previously installed models from your device.</div>""")
                clear_models_button = gr.Button("Remove models from memory", variant="primary")
            with gr.Column(variant="panel"):
                output_dir = gr.Textbox(value=default_output_dir, label="Output Directory", info="Specify the path to save output files.", placeholder="output/UVR_output")

        with gr.Accordion("Rename Stems", open=False):
            gr.Markdown(
                """
                Use keys to automatically format output file names.

                Available keys:
                * **NAME** - Input file name
                * **STEM** - Stem type (e.g., Vocals, Instrumental)
                * **MODEL** - Model name (e.g., BS-Roformer-Viperx-1297)

                > Example:
                > * **Template:** NAME_(STEM)_MODEL
                > * **Result:** Music_(Vocals)_BS-Roformer-Viperx-1297
                """
            )
            with gr.Row():
                vocals_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Vocal Stem", info="Example: Music_(Vocals)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                instrumental_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Instrumental Stem", info="Example: Music_(Instrumental)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                other_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Other Stem", info="Example: Music_(Other)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
            with gr.Row():
                drums_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Drum Stem", info="Example: Music_(Drums)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                bass_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Bass Stem", info="Example: Music_(Bass)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
            with gr.Row():
                guitar_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Guitar Stem", info="Example: Music_(Guitar)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                piano_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Piano Stem", info="Example: Music_(Piano)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")

    with gr.Tab("Leaderboard"):
        with gr.Group():
            with gr.Row(equal_height=True):
                list_filter = gr.Dropdown(value="vocals", choices=["vocals", "instrumental", "drums", "bass", "guitar", "piano", "other"], label="Filter", info="Filter models by stem type.")
                list_limit = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Limit", info="Limit the number of displayed models.")
                list_button = gr.Button("Refresh List", variant="primary")

        output_list = gr.HTML(label="Leaderboard")

    roformer_override_seg_size.change(show_hide_params, inputs=[roformer_override_seg_size], outputs=[roformer_seg_size])
    mdx23c_override_seg_size.change(show_hide_params, inputs=[mdx23c_override_seg_size], outputs=[mdx23c_seg_size])
    vr_post_process.change(show_hide_params, inputs=[vr_post_process], outputs=[vr_post_process_threshold])

    demucs_model.change(update_stems, inputs=[demucs_model], outputs=stem6)

    list_button.click(leaderboard, inputs=[list_filter, list_limit], outputs=output_list)

    clear_models_button.click(clear_models, inputs=[model_file_dir])

    roformer_button.click(
        roformer_separator,
        inputs=[
            roformer_audio,
            roformer_model,
            roformer_seg_size,
            roformer_override_seg_size,
            roformer_overlap,
            roformer_pitch_shift,
            model_file_dir,
            output_dir,
            roformer_output_format,
            roformer_norm_threshold,
            roformer_amp_threshold,
            roformer_batch_size,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[roformer_stem1, roformer_stem2],
    )
    mdx23c_button.click(
        mdx23c_separator,
        inputs=[
            mdx23c_audio,
            mdx23c_model,
            mdx23c_seg_size,
            mdx23c_override_seg_size,
            mdx23c_overlap,
            mdx23c_pitch_shift,
            model_file_dir,
            output_dir,
            mdx23c_output_format,
            mdx23c_norm_threshold,
            mdx23c_amp_threshold,
            mdx23c_batch_size,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx23c_stem1, mdx23c_stem2],
    )
    mdx_button.click(
        mdx_separator,
        inputs=[
            mdx_audio,
            mdx_model,
            mdx_hop_length,
            mdx_seg_size,
            mdx_overlap,
            mdx_denoise,
            model_file_dir,
            output_dir,
            mdx_output_format,
            mdx_norm_threshold,
            mdx_amp_threshold,
            mdx_batch_size,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx_stem1, mdx_stem2],
    )
    vr_button.click(
        vr_separator,
        inputs=[
            vr_audio,
            vr_model,
            vr_window_size,
            vr_aggression,
            vr_tta,
            vr_post_process,
            vr_post_process_threshold,
            vr_high_end_process,
            model_file_dir,
            output_dir,
            vr_output_format,
            vr_norm_threshold,
            vr_amp_threshold,
            vr_batch_size,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[vr_stem1, vr_stem2],
    )
    demucs_button.click(
        demucs_separator,
        inputs=[
            demucs_audio,
            demucs_model,
            demucs_seg_size,
            demucs_shifts,
            demucs_overlap,
            demucs_segments_enabled,
            model_file_dir,
            output_dir,
            demucs_output_format,
            demucs_norm_threshold,
            demucs_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[demucs_stem1, demucs_stem2, demucs_stem3, demucs_stem4, demucs_stem5, demucs_stem6],
    )

def main():
    with gr.Blocks(
        title="🎵 PolUVR 🎵",
        css="footer{display:none !important}",
        theme=gr.themes.Default(spacing_size="sm", radius_size="lg")
    ) as app:
        gr.HTML("<h1><center> 🎵 PolUVR 🎵 </center></h1>")
        PolUVR_UI()

    app.queue().launch(
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        debug=True,
        show_error=True,
    )

if __name__ == "__main__":
    main()
