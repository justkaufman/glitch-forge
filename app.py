import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import random
import math

st.set_page_config(page_title="GLITCH FORGE", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
.stApp { background-color: #0a0a0a; }
h1, h2, h3, p, span, label, .stMarkdown { color: #e0e0e0 !important; font-family: 'Space Mono', monospace !important; }
section[data-testid="stSidebar"] { background-color: #f0f0f0 !important; border-right: 1px solid #cccccc !important; }
section[data-testid="stSidebar"] * { color: #222222 !important; font-family: 'Space Mono', monospace !important; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: #222222 !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }
section[data-testid="stSidebar"] h3 { color: #000000 !important; font-size: 13px !important; font-weight: 700 !important; letter-spacing: 3px; }
[data-testid="stDownloadButton"] button, [data-testid="stButton"] button {
  background-color: #ff3366 !important;
  color: #ffffff !important;
  border: none !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  padding: 0.6rem 2rem !important;
}
[data-testid="stDownloadButton"] button:hover, [data-testid="stButton"] button:hover {
  background-color: #cc0044 !important;
  color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# ░▒▓ GLITCH FORGE ▓▒░")
st.caption("Brutalist ad visuals. Animated glitch loops. Export PNG or GIF.")

# --- Engine ---
def make_gradient(w, h, colors, angle=0):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    denom = abs(w * cos_a) + abs(h * sin_a)
    if denom == 0:
        denom = 1
    for y in range(h):
        for x in range(w):
            t = (x * cos_a + y * sin_a) / denom
            t = max(0.0, min(1.0, t))
            idx = t * (len(colors) - 1)
            i = int(idx)
            f = idx - i
            if i >= len(colors) - 1:
                i, f = len(colors) - 2, 1.0
            c = [int(colors[i][j] * (1 - f) + colors[i + 1][j] * f) for j in range(3)]
            img[y, x] = c
    return Image.fromarray(img)

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def channel_shift(img, amount):
    arr = np.array(img)
    arr[:, :, 0] = np.roll(arr[:, :, 0], amount, axis=1)
    arr[:, :, 2] = np.roll(arr[:, :, 2], -amount, axis=1)
    return Image.fromarray(arr)

def scan_lines(img, density, alpha):
    arr = np.array(img).astype(np.float32)
    for y in range(0, arr.shape[0], max(1, density)):
        arr[y, :, :] *= (1.0 - alpha)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def pixel_sort(img, threshold, vertical=False):
    arr = np.array(img)
    if vertical:
        arr = np.transpose(arr, (1, 0, 2))
    for y in range(arr.shape[0]):
        row = arr[y]
        brightness = np.mean(row, axis=1)
        mask = brightness > threshold
        regions = []
        start = None
        for x in range(len(mask)):
            if mask[x] and start is None:
                start = x
            elif not mask[x] and start is not None:
                regions.append((start, x))
                start = None
        if start is not None:
            regions.append((start, len(mask)))
        for s, e in regions:
            segment = row[s:e]
            indices = np.argsort(np.mean(segment, axis=1))
            arr[y, s:e] = segment[indices]
    if vertical:
        arr = np.transpose(arr, (1, 0, 2))
    return Image.fromarray(arr)

def block_glitch(img, intensity, seed):
    rng = random.Random(seed)
    arr = np.array(img)
    h, w = arr.shape[:2]
    for _ in range(intensity):
        bh = rng.randint(2, max(3, h // 8))
        bw = rng.randint(w // 4, w)
        y = rng.randint(0, h - bh)
        x = rng.randint(0, w - bw)
        shift = rng.randint(-w // 4, w // 4)
        block = arr[y:y+bh, x:x+bw].copy()
        new_x = max(0, min(w - block.shape[1], x + shift))
        arr[y:y+bh, new_x:new_x+block.shape[1]] = block[:, :min(block.shape[1], w - new_x)]
    return Image.fromarray(arr)

def add_noise(img, amount):
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-amount, amount, arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def add_vignette(img):
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vign = 1.0 - (dist / max_dist) ** 1.5 * 0.7
    arr *= vign[:, :, np.newaxis]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def add_text_brutal(img, headline, subtext, font_color, glitch_text, text_offset=0):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    padding = w // 16
    try:
        font_path_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font_path_mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        font_size = w // 8
        font_lg = ImageFont.truetype(font_path_bold, font_size)
        # Shrink until headline fits within canvas width
        while font_size > 10:
            font_lg = ImageFont.truetype(font_path_bold, font_size)
            bbox_test = draw.textbbox((0, 0), headline, font=font_lg)
            if (bbox_test[2] - bbox_test[0]) <= w - padding * 2:
                break
            font_size -= 2
        font_sm = ImageFont.truetype(font_path_mono, w // 22)
    except:
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), headline, font=font_lg)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = (w - tw) // 2, int(h * 0.35)

    if glitch_text:
        off = 4 + text_offset
        for dx, dy, color in [(off, 2, "#ff0044"), (-off, -2, "#00ffcc")]:
            draw.text((tx + dx, ty + dy), headline, fill=color, font=font_lg)

    draw.text((tx, ty), headline, fill=font_color, font=font_lg)

    bbox2 = draw.textbbox((0, 0), subtext, font=font_sm)
    tw2 = bbox2[2] - bbox2[0]
    draw.text(((w - tw2) // 2, ty + th + 30), subtext, fill=font_color, font=font_sm)
    return img

def render_frame(base_img, frame_idx, cfg):
    img = base_img.copy()
    seed_offset = frame_idx * 7

    if cfg["do_pixelsort"]:
        thresh = cfg["sort_thresh"] + int(math.sin(frame_idx * 0.5) * 30)
        img = pixel_sort(img, max(0, min(255, thresh)), cfg["sort_vert"])
    if cfg["do_block"]:
        img = block_glitch(img, cfg["block_int"], cfg["block_seed"] + seed_offset)
    if cfg["do_channel"]:
        shift = cfg["channel_amt"] + int(math.sin(frame_idx * 0.8) * cfg["channel_amt"] * 0.6)
        img = channel_shift(img, shift)
    if cfg["do_scanlines"]:
        offset_density = max(1, cfg["scan_density"] + (frame_idx % 3) - 1)
        img = scan_lines(img, offset_density, cfg["scan_alpha"])
    if cfg["do_noise"]:
        img = add_noise(img, cfg["noise_amt"])
    if cfg["do_vignette"]:
        img = add_vignette(img)

    text_jitter = int(math.sin(frame_idx * 1.2) * 3) if cfg["anim_text_jitter"] else 0
    img = add_text_brutal(img, cfg["headline"], cfg["subtext"], cfg["font_color"], cfg["glitch_text"], text_jitter)
    return img

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🔧 CANVAS")
    width = st.select_slider("Width", [400, 600, 800, 1080], value=600)
    height = st.select_slider("Height", [400, 600, 800, 1080], value=600)
    col1 = st.color_picker("Gradient Start", "#0d0028")
    col2 = st.color_picker("Gradient End", "#1a003a")
    col3 = st.color_picker("Accent Mid", "#220044")
    grad_angle = st.slider("Gradient Angle", 0, 360, 135)

    st.markdown("### ✏️ COPY")
    headline = st.text_input("Headline", "BREAK THE FEED")
    subtext = st.text_input("Subtext", "NEW DROP — 03.01.26")
    font_color = st.color_picker("Text Color", "#ffffff")
    glitch_text = st.checkbox("Glitch text offset", value=True)

    st.markdown("### 📡 DISTORTION")
    do_channel = st.checkbox("Channel Shift", value=True)
    channel_amt = st.slider("Shift Amount", 1, 40, 12)
    do_scanlines = st.checkbox("Scan Lines", value=True)
    scan_density = st.slider("Line Density", 1, 8, 3)
    scan_alpha = st.slider("Line Opacity", 0.0, 1.0, 0.35)
    do_pixelsort = st.checkbox("Pixel Sort", value=False)
    sort_thresh = st.slider("Sort Threshold", 0, 255, 120)
    sort_vert = st.checkbox("Vertical Sort", value=False)
    do_block = st.checkbox("Block Glitch", value=True)
    block_int = st.slider("Block Intensity", 1, 30, 8)
    block_seed = st.number_input("Seed", value=42, step=1)
    do_noise = st.checkbox("Film Grain", value=True)
    noise_amt = st.slider("Grain Amount", 1, 60, 20)
    do_vignette = st.checkbox("Vignette", value=True)

    st.markdown("### 🎬 ANIMATION")
    export_mode = st.radio("Export Mode", ["PNG (still)", "GIF (animated)"], horizontal=True)
    num_frames = st.slider("Frames", 4, 30, 12)
    frame_duration = st.slider("Frame duration (ms)", 50, 500, 120)
    anim_text_jitter = st.checkbox("Animate text jitter", value=True)
    gif_loop = st.checkbox("Loop forever", value=True)

# --- Config dict ---
cfg = dict(
    do_channel=do_channel, channel_amt=channel_amt,
    do_scanlines=do_scanlines, scan_density=scan_density, scan_alpha=scan_alpha,
    do_pixelsort=do_pixelsort, sort_thresh=sort_thresh, sort_vert=sort_vert,
    do_block=do_block, block_int=block_int, block_seed=block_seed,
    do_noise=do_noise, noise_amt=noise_amt, do_vignette=do_vignette,
    headline=headline, subtext=subtext, font_color=font_color,
    glitch_text=glitch_text, anim_text_jitter=anim_text_jitter,
)

# --- Render base ---
colors = [hex_to_rgb(col1), hex_to_rgb(col3), hex_to_rgb(col2)]
base_img = make_gradient(width, height, colors, grad_angle)

if export_mode == "PNG (still)":
    final = render_frame(base_img, 0, cfg)
    st.image(final, use_container_width=True)
    buf = io.BytesIO()
    final.save(buf, format="PNG")
    st.download_button("⬇ EXPORT PNG", buf.getvalue(), file_name="glitch_forge.png", mime="image/png")

else:
    with st.spinner("Rendering frames..."):
        frames = [render_frame(base_img, i, cfg) for i in range(num_frames)]

    preview_buf = io.BytesIO()
    frames[0].save(
        preview_buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=frame_duration, loop=0 if gif_loop else 1, optimize=True
    )
    st.image(preview_buf.getvalue(), use_container_width=True)

    dl_buf = io.BytesIO()
    frames[0].save(
        dl_buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=frame_duration, loop=0 if gif_loop else 1, optimize=True
    )
    st.download_button("⬇ EXPORT GIF", dl_buf.getvalue(), file_name="glitch_forge.gif", mime="image/gif")

    st.caption(f"{num_frames} frames · {frame_duration}ms/frame · ~{num_frames * frame_duration / 1000:.1f}s loop")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#333; font-size:11px; letter-spacing:3px;'>GLITCH FORGE — MADE FOR THOSE WHO DON'T DO SAFE</p>",
    unsafe_allow_html=True,
)
