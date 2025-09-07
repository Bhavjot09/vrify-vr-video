import os
import sys
import tempfile
import subprocess
import streamlit as st
import cv2
import numpy as np


# 🔍 Debug info
print("👉 Streamlit is running with Python:", sys.executable)


# ============= Core VR conversion =============

def warp_horizontal_curve(img: np.ndarray, curve: float) -> np.ndarray:
    """Apply simple horizontal curve distortion."""
    if curve <= 0:
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

    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT
    )


def convert_to_vr(input_path: str, output_path: str,
                  layout: str = "sbs", curve: float = 0.0,
                  width: int = 1920, height: int = 1080) -> None:
    """Convert video into VR style (SBS or OU) using OpenCV."""

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if layout == "sbs":
        pane_w, pane_h = width // 2, height
    else:  # over-under
        pane_w, pane_h = width, height // 2

    temp_video_path = output_path.replace(".mp4", "_noaudio.mp4")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (pane_w, pane_h), interpolation=cv2.INTER_AREA)

        if curve > 0:
            frame = warp_horizontal_curve(frame, curve)

        if layout == "sbs":
            vr_frame = np.hstack([frame, frame])
        else:
            vr_frame = np.vstack([frame, frame])

        out.write(vr_frame)

    cap.release()
    out.release()

    # 🔊 Merge audio with FFmpeg
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_video_path,
            "-i", input_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(temp_video_path)  # clean up
    except Exception as e:
        print("⚠️ Audio merge failed:", e)
        os.rename(temp_video_path, output_path)  # fallback: video only


# ============= Streamlit UI =============

st.set_page_config(page_title="VRify App (OpenCV + Audio)", page_icon="🎥", layout="centered")
st.title("🎥 VRify – Convert a video to VR (SBS / Over-Under) [OpenCV + FFmpeg Audio]")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

layout = st.radio("Select VR Layout", ["sbs", "ou"], format_func=lambda x: "Side-by-Side" if x == "sbs" else "Over-Under")
width = st.number_input("Output Width", min_value=640, max_value=7680, step=2, value=1920)
height = st.number_input("Output Height", min_value=480, max_value=4320, step=2, value=1080)
curve = st.slider("Curve distortion", 0.0, 0.5, 0.0, 0.05)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    output_path = os.path.join(tempfile.gettempdir(), "vrified_video.mp4")

    st.write("⚙️ Processing... please wait")
    try:
        convert_to_vr(input_path, output_path, layout=layout, curve=curve, width=width, height=height)
        st.success("✅ Conversion complete with audio!")
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download VR Video", f, file_name="vrified_video.mp4", mime="video/mp4")
    except Exception as e:
        st.error(f"❌ Error: {e}")
