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
# PATHS (STREAMLIT SAFE)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
FONT_PATH = os.path.join(BASE_DIR, "FE-FONT.TTF")

# =========================
# IMAGE UTIL
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
# IMPROVED THRESHOLD (IMPORTANT FIX)
# =========================
def preprocess(gray):
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return th

# =========================
# CHARACTER EXTRACTION (FIXED)
# =========================
def extract_characters(img, num_chars=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = preprocess(gray)

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 15 and w > 5:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    # SAFE fallback (IMPORTANT FIX)
    h_img, w_img = th.shape
    if len(boxes) < num_chars:
        cw = max(1, w_img // num_chars)
        boxes = [(i * cw, 0, cw, h_img) for i in range(num_chars)]

    boxes = boxes[:num_chars]

    chars = []
    for x, y, w, h in boxes:
        char = th[y:y+h, x:x+w]
        char = square_pad_and_resize(char)
        chars.append(char)

    return chars, th

# =========================
# CONTOUR POINTS
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
# TEMPLATE GENERATION
# =========================
def generate_templates():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    if not os.path.exists(FONT_PATH):
        st.error("FE-FONT.TTF missing in project root!")
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

    st.success("Templates generated!")

# =========================
# ENCODE FIX (MAIN FIX)
# =========================
if mode == "Encode Plate":
    st.title("Encode Plate")

    file = st.file_uploader("Upload image")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        chars, th = extract_characters(img)

        st.image(th, caption="Threshold Output")

        encoded = []

        for c in chars:
            pts = extract_contour_points(c)

            # ignore empty junk
            if np.count_nonzero(pts) == 0:
                continue

            encoded.append({
                "points": pts.tolist()
            })

        # IMPORTANT FIX: validation
        if len(encoded) == 0:
            st.error("No characters detected. Try a clearer image.")
        else:
            st.success(f"Encoded {len(encoded)} characters")
            st.code(json.dumps(encoded, indent=2))

# =========================
# TEMPLATE VIEW / DEBUG
# =========================
elif mode == "Generate Templates":
    st.title("Template Generator")

    if st.button("Generate Templates"):
        generate_templates()

# =========================
# PLACEHOLDER
# =========================
elif mode == "Decode & Recognize":
    st.title("Decode & Recognize")
    st.info("Your recognition logic stays same (not modified here).")