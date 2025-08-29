import streamlit as st
import os
import tempfile
import subprocess

# =============================
# VRIFY CORE LOGIC (merged from vrify.py)
# =============================
import shutil
import sys
from typing import Optional, Tuple

try:
    import cv2
    import numpy as np
    from moviepy.editor import VideoFileClip
except Exception:
    cv2 = None
    np = None
    VideoFileClip = None


def which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def ensure_dims(input_path: str, layout: str, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    if width and height:
        return width, height
    if layout == "sbs":
        return width or 3840, height or 1080
    else:
        return width or 1920, height or 2160


def build_ffmpeg_filter(layout: str, pad: int, keep_ar: bool, out_w: int, out_h: int) -> str:
    if layout == "sbs":
        pane_w = out_w // 2
        pane_h = out_h
        stacker = f"hstack=inputs=2"
    else:
        pane_w = out_w
        pane_h = out_h // 2
        stacker = f"vstack=inputs=2"

    if keep_ar:
        scale = f"scale=w='if(gte(a,{pane_w}/{pane_h}),{pane_w},-2)':h='if(gte(a,{pane_w}/{pane_h}),-2,{pane_h})',pad={pane_w}:{pane_h}:(ow-iw)/2:(oh-ih)/2"
    else:
        scale = f"scale={pane_w}:{pane_h}"

    if pad > 0:
        inner_w = pane_w - 2 * pad
        inner_h = pane_h - 2 * pad
        scale = f"{scale},scale={inner_w}:{inner_h},pad={pane_w}:{pane_h}:{pad}:{pad}"

    filtergraph = f"[0:v]{scale}[pane];[pane][pane]{stacker}"
    return filtergraph


def run_ffmpeg(input_path: str, output_path: str, layout: str, out_w: int, out_h: int, fps: Optional[float], crf: int, bitrate: Optional[str], pad: int, keep_ar: bool) -> None:
    ffmpeg_bin = which_ffmpeg()
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found in PATH")

    vf = build_ffmpeg_filter(layout, pad, keep_ar, out_w, out_h)

    cmd = [ffmpeg_bin, "-y", "-i", input_path]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += [
        "-filter_complex", vf,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
    ]
    if bitrate:
        cmd += ["-b:v", bitrate]
    else:
        cmd += ["-crf", str(crf), "-preset", "medium"]

    cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-vf", f"crop=iw-((iw)%2):ih-((ih)%2)"]
    cmd += [output_path]

    subprocess.run(cmd, check=True)


def warp_horizontal_curve(img, curve: float) -> "np.ndarray":
    if curve <= 0 or cv2 is None or np is None:
        return img

    h, w = img.shape[:2]
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    k = curve * 0.25
    r2 = xv ** 2
    x_distort = xv * (1 + k * r2)
    y_distort = yv

    map_x = ((x_distort + 1) * 0.5) * (w - 1)
    map_y = ((y_distort + 1) * 0.5) * (h - 1)

    curved = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return curved


def python_stack(input_path: str, output_path: str, layout: str, out_w: int, out_h: int, fps: Optional[float], pad: int, curve: float, keep_ar: bool):
    if VideoFileClip is None:
        raise RuntimeError("MoviePy/OpenCV not installed. pip install moviepy opencv-python numpy")

    clip = VideoFileClip(input_path)
    if fps is None:
        fps = clip.fps

    if layout == "sbs":
        pane_w, pane_h = out_w // 2, out_h
    else:
        pane_w, pane_h = out_w, out_h // 2

    def fit_frame(frame):
        h, w = frame.shape[:2]
        if keep_ar:
            scale = min((pane_w - 2*pad) / w, (pane_h - 2*pad) / h)
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
            x0 = (pane_w - new_w) // 2
            y0 = (pane_h - new_h) // 2
            canvas[y0:y0+new_h, x0:x0+new_w] = resized
        else:
            canvas = cv2.resize(frame, (pane_w, pane_h), interpolation=cv2.INTER_AREA)

        if curve > 0:
            canvas = warp_horizontal_curve(canvas, curve)
        return canvas

    def make_frame(t):
        frame = clip.get_frame(t)
        left = fit_frame(frame)
        right = left
        if layout == "sbs":
            out = np.hstack([left, right])
        else:
            out = np.vstack([left, right])
        return out

    out_clip = clip.fl(make_frame, apply_to=["mask"]).set_duration(clip.duration)
    out_clip = out_clip.set_fps(fps)
    if clip.audio is not None:
        out_clip = out_clip.set_audio(clip.audio)

    out_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        temp_audiofile="_vrify_aud.m4a",
        remove_temp=True,
        preset="medium",
        threads=os.cpu_count() or 4,
    )

# =============================
# STREAMLIT UI
# =============================

st.set_page_config(page_title="VRify App", page_icon="🎥", layout="centered")
st.title("🎥 VRify – Convert any video to VR SBS/OU format")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi"])

layout = st.radio("Layout", ["sbs", "ou"], format_func=lambda x: "Side-by-Side" if x=="sbs" else "Over/Under")
width = st.number_input("Output Width", min_value=640, max_value=7680, step=2, value=3840 if layout=="sbs" else 1920)
height = st.number_input("Output Height", min_value=480, max_value=4320, step=2, value=1080 if layout=="sbs" else 2160)
curve = st.slider("Curve", 0.0, 0.5, 0.0, 0.05)
pad = st.number_input("Padding (px)", min_value=0, max_value=200, value=0)

col1, col2 = st.columns(2)
with col1:
    use_ffmpeg = st.checkbox("Use ffmpeg if available", value=True)
with col2:
    keep_ar = st.checkbox("Preserve aspect ratio", value=True)

if uploaded_file is not None:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.flush()

    output_name = f"output_{layout.upper()}.mp4"
    temp_output = os.path.join(tempfile.gettempdir(), output_name)

    if st.button("Convert to VR video"):
        st.info("Processing... this may take a while depending on video size.")
        out_w, out_h = ensure_dims(temp_input.name, layout, width, height)

        try:
            if use_ffmpeg and which_ffmpeg():
                run_ffmpeg(temp_input.name, temp_output, layout, out_w, out_h, fps=None, crf=18, bitrate=None, pad=pad, keep_ar=keep_ar)
            else:
                python_stack(temp_input.name, temp_output, layout, out_w, out_h, fps=None, pad=pad, curve=curve, keep_ar=keep_ar)
            st.success("✅ Done! Your VR video is ready.")

            with open(temp_output, "rb") as f:
                st.download_button("Download VR Video", f, file_name=output_name, mime="video/mp4")

        except Exception as e:
            st.error(f"Conversion failed: {e}")
            
            