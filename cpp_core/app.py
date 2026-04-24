import os
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# =========================
# CONFIG
# =========================
NUM_POINTS = 40

st.set_page_config(page_title="Number Plate AI", layout="wide")

mode = st.sidebar.selectbox(
    "Choose mode",
    ["Generate Templates", "Encode Plate", "Decode & Recognize"]
)

# =========================
# SAFE PATHS (IMPORTANT FIX)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
FONT_PATH = os.path.join(BASE_DIR, "FE-FONT.TTF")

# =========================
# IMAGE HELPERS
# =========================
def square_pad_and_resize(img, size=(40, 40)):
    h, w = img.shape[:2]
    side = max(h, w)

    pad_y = (side - h) // 2
    pad_x = (side - w) // 2

    padded = cv2.copyMakeBorder(
        img,
        pad_y, side - h - pad_y,
        pad_x, side - w - pad_x,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return cv2.resize(padded, size)

# =========================
# LAGRANGE INTERPOLATION
# =========================
def lagrange_1d(t, y, t_new):
    t = np.array(t, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    result = np.zeros_like(t_new, dtype=np.float64)
    n = len(t)

    for i in range(n):
        Li = np.ones_like(t_new)

        for j in range(n):
            if i != j:
                denom = (t[i] - t[j])
                if abs(denom) < 1e-12:
                    continue
                Li *= (t_new - t[j]) / denom

        result += y[i] * Li

    return result


def lagrange_smooth(pts, num_out=40):
    pts = np.array(pts, dtype=np.float64)

    if len(pts) < 4:
        return pts

    t = np.linspace(0, 1, len(pts))
    t_new = np.linspace(0, 1, num_out)

    x = pts[:, 0]
    y = pts[:, 1]

    x_new = lagrange_1d(t, x, t_new)
    y_new = lagrange_1d(t, y, t_new)

    return np.vstack([x_new, y_new]).T

# =========================
# CURVE CHECK
# =========================
def is_curved(pts):
    pts = np.array(pts)
    diffs = np.diff(pts, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.std(angles) > 0.6

# =========================
# TEMPLATE GENERATION
# =========================
def generate_templates():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    if not os.path.exists(FONT_PATH):
        st.error("FE-FONT.TTF not found in project root!")
        return

    font = ImageFont.truetype(FONT_PATH, 90)

    for ch in chars:
        img = Image.new("L", (150, 150), 0)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.text(
            ((150 - w) // 2, (150 - h) // 2),
            ch,
            font=font,
            fill=255
        )

        img = np.array(img)

        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = square_pad_and_resize(img[y:y+h, x:x+w])

        cv2.imwrite(os.path.join(TEMPLATE_DIR, f"{ch}.png"), img)

    st.success("Templates generated successfully!")

# =========================
# CONTOUR EXTRACTION
# =========================
def extract_contour_points(img, num_points=NUM_POINTS):
    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return np.zeros((num_points, 2))

    cnt = max(contours, key=cv2.contourArea)[:, 0, :]
    idx = np.linspace(0, len(cnt)-1, num_points).astype(int)

    return cnt[idx].astype(np.float64)

# =========================
# NORMALIZATION
# =========================
def normalize_points(pts):
    pts = pts - np.mean(pts, axis=0)
    norm = np.linalg.norm(pts)
    return pts / norm if norm > 1e-9 else pts

# =========================
# LOAD TEMPLATES
# =========================
@st.cache_data
def load_templates():
    templates = {}

    if not os.path.exists(TEMPLATE_DIR):
        return templates

    for f in os.listdir(TEMPLATE_DIR):
        path = os.path.join(TEMPLATE_DIR, f)

        img = cv2.imread(path, 0)
        if img is None:
            continue

        img = cv2.resize(img, (40, 40))

        pts = extract_contour_points(img)
        pts = normalize_points(pts)

        templates[f[0]] = pts

    return templates

# =========================
# DISTANCE
# =========================
def curve_distance(a, b):
    return np.mean(np.linalg.norm(a - b, axis=1))

# =========================
# RECOGNITION
# =========================
def recognize(pts, templates):
    pts = normalize_points(pts)

    if is_curved(pts):
        pts = lagrange_smooth(pts)

    best, best_score = "?", float("inf")

    for label, tpl in templates.items():
        score = curve_distance(pts, tpl)

        if score < best_score:
            best_score = score
            best = label

    return best

# =========================
# UI
# =========================
if mode == "Generate Templates":
    st.title("Template Generator")

    if st.button("Generate Templates"):
        generate_templates()

elif mode == "Encode Plate":
    st.title("Encode Plate")

    file = st.file_uploader("Upload image")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        st.image(th)

elif mode == "Decode & Recognize":
    st.title("Decode & Recognize")

    templates = load_templates()

    text = st.text_area("Paste JSON")

    if st.button("Recognize"):
        if not text.strip():
            st.warning("Please paste JSON data")
        else:
            data = json.loads(text)

            result = ""

            for ch in data:
                pts = np.array(ch["points"])
                result += recognize(pts, templates)

            st.success(f"RESULT: {result}")