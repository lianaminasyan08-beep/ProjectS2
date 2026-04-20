import cv2
import numpy as np
import os

# -----------------------------
# LOAD TEMPLATES (A-Z, 0-9)
# -----------------------------
def load_templates(path="templates"):
    templates = {}

    for file in os.listdir(path):
        label = file.split(".")[0]   # A.png → A

        img = cv2.imread(os.path.join(path, file), 0)
        img = cv2.resize(img, (60, 60))

        templates[label] = img

    return templates


# -----------------------------
# EXTRACT CHAR SEGMENTS
# -----------------------------
def segment_characters(img):
    h, w = img.shape

    # simple fixed segmentation (you can upgrade later)
    step = w // 7   # plate assumed max 7 chars

    chars = []

    for i in range(7):
        part = img[:, i*step:(i+1)*step]
        part = cv2.resize(part, (60, 60))
        chars.append(part)

    return chars


# -----------------------------
# RECOGNITION (ROBUST)
# -----------------------------
def recognize_plate(image_path):
    img = cv2.imread(image_path, 0)

    if img is None:
        return "ERROR"

    img = cv2.resize(img, (700, 200))

    templates = load_templates()

    segments = segment_characters(img)

    result = ""

    for seg in segments:

        best_label = "?"
        best_score = float("inf")

        for label, tpl in templates.items():

            # ORB feature matching (robust)
            orb = cv2.ORB_create()

            kp1, des1 = orb.detectAndCompute(seg, None)
            kp2, des2 = orb.detectAndCompute(tpl, None)

            if des1 is None or des2 is None:
                continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)

            score = sum([m.distance for m in matches])

            if score < best_score:
                best_score = score
                best_label = label

        result += best_label

    return result