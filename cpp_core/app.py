import os
import json
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
NUM_POINTS = 40

mode = st.sidebar.selectbox(
    "Choose mode",
    ["Generate Templates", "Encode Plate", "Decode & Recognize"]
)

# =========================
# HELPERS
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
# TEMPLATE GENERATION
# =========================
def generate_templates():
    os.makedirs("templates", exist_ok=True)

    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    font_path = os.path.join(os.getcwd(), "FE-FONT.TTF")

    if not os.path.exists(font_path):
        st.error("Font FE-FONT.TTF not found!")
        return

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, 90)

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
            x, y, wc, hc = cv2.boundingRect(coords)
            img = square_pad_and_resize(img[y:y + hc, x:x + wc])

        cv2.imwrite(f"templates/{ch}.png", img)

    st.success("Templates generated!")

# =========================
# CHARACTER EXTRACTION
# =========================
def extract_characters(img, num_chars=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = [
        cv2.boundingRect(c)
        for c in contours
        if cv2.boundingRect(c)[3] > 15
    ]

    boxes = sorted(boxes, key=lambda b: b[0])

    merged = []
    for b in boxes:
        if not merged:
            merged.append(b)
        else:
            x, y, w, h = b
            mx, my, mw, mh = merged[-1]

            if x < mx + mw + 5:
                merged[-1] = (
                    min(mx, x),
                    min(my, y),
                    max(mx + mw, x + w) - min(mx, x),
                    max(my + mh, y + h) - min(my, y)
                )
            else:
                merged.append(b)

    if len(merged) < num_chars:
        h_img, w_img = th.shape
        cw = w_img // num_chars
        merged = [(i * cw, 0, cw, h_img) for i in range(num_chars)]

    merged = merged[:num_chars]

    chars = []
    for x, y, w, h in merged:
        char = th[y:y + h, x:x + w]
        char = square_pad_and_resize(char)
        chars.append(char)

    return chars, th, merged

# =========================
# CONTOUR POINT EXTRACTION (FIXED)
# =========================
def extract_contour_points(img, num_points=NUM_POINTS):
    img = np.array(img, dtype=np.uint8)

    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return np.zeros((num_points, 2))

    cnt = max(contours, key=cv2.contourArea)

    if len(cnt) < num_points:
        cnt = cv2.convexHull(cnt)

    cnt = cnt[:, 0, :]

    idx = np.linspace(0, len(cnt) - 1, num_points).astype(int)

    return cnt[idx].astype(float)

# =========================
# NORMALIZATION (FIXED STABILITY)
# =========================
def normalize_points(pts):
    pts = np.array(pts, dtype=np.float64)

    pts -= np.mean(pts, axis=0)

    norm = np.linalg.norm(pts)
    if norm > 1e-9:
        pts /= norm

    return pts

# =========================
# LAGRANGE (SAFE VISUAL ONLY)
# =========================
def lagrange(x, y, t):
    n = len(x)
    result = 0.0

    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                denom = x[i] - x[j]
                if abs(denom) < 1e-9:
                    continue
                term *= (t - x[j]) / denom
        result += term

    return result

# =========================
# SHAPE MATCHING
# =========================
def shape_distance(a, b):
    return np.mean(np.linalg.norm(a - b, axis=1))

# =========================
# ORDER POINTS (FIX STRAIGHT LINE BUG)
# =========================
def order_points(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    return pts[np.argsort(angles)]

# =========================
# LOAD TEMPLATES
# =========================
@st.cache_data
def load_templates():
    pts_dict = {}
    img_dict = {}

    for f in os.listdir("templates"):
        img = cv2.imread(f"templates/{f}", 0)
        img = cv2.resize(img, (40, 40))

        pts = extract_contour_points(img)
        pts = normalize_points(pts)

        label = f[0]

        pts_dict[label] = pts
        img_dict[label] = img

    return pts_dict, img_dict

# =========================
# RECOGNITION
# =========================
def recognize(pts, templates):
    best = "?"
    best_score = 1e9

    for label, tpl in templates.items():
        score = shape_distance(pts, tpl)

        if score < best_score:
            best_score = score
            best = label

    return best

# =========================
# UI
# =========================

if mode == "Generate Templates":
    st.title("Template Generator")

    if st.button("Generate"):
        generate_templates()

# -------------------------
# ENCODER (FIXED)
# -------------------------
elif mode == "Encode Plate":
    file = st.file_uploader("Upload image")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        chars, th, boxes = extract_characters(img)

        st.image(th)

        encoded = []

        for c in chars:
            pts = extract_contour_points(c)
            encoded.append({
                "points": pts.tolist()
            })

        json_data = json.dumps(encoded, indent=2)

        st.code(json_data)

        st.download_button(
            "Download JSON",
            json_data,
            file_name="encoded.json",
            mime="application/json"
        )

# -------------------------
# DECODER
# -------------------------
elif mode == "Decode & Recognize":
    templates_pts, templates_imgs = load_templates()

    text = st.text_area("Paste JSON")

    if st.button("Recognize") and text.strip():

        data = json.loads(text)

        result = ""

        for i, ch in enumerate(data):

            pts = np.array(ch["points"])
            pts = normalize_points(pts)

            pts = order_points(pts)

            label = recognize(pts, templates_pts)

            result += label

            # SAFE PLOT (NO DISTORTION)
            center = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:,1]-center[1], pts[:,0]-center[0])
            pts = pts[np.argsort(angles)]

            x = pts[:, 0]
            y = pts[:, 1]

            if len(x) > 5:
                xi = np.linspace(min(x), max(x), 100)
                yi = [lagrange(x, y, v) for v in xi]

                fig, ax = plt.subplots()
                ax.plot(xi, yi)
                ax.set_title(f"Char {i} → {label}")
                ax.invert_yaxis()

                st.pyplot(fig)

        st.success(f"RESULT: {result}")