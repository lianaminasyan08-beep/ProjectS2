import numpy as np
import matplotlib.pyplot as plt

# ---------------- LAGRANGE ----------------
def lagrange(x, y, xi):
    total = 0
    n = len(x)

    for i in range(n):
        term = y[i]

        for j in range(n):
            if i != j:
                denom = x[i] - x[j]
                if abs(denom) < 1e-9:
                    continue
                term *= (xi - x[j]) / denom

        total += term

    return total


# ---------------- SAFE PLOT ----------------
def plot_points(pts):

    pts = np.array(pts, dtype=np.float64)

    if len(pts) < 3:
        return

    x = pts[:, 0]
    y = pts[:, 1]

    # normalize (VERY IMPORTANT for stability)
    x = x - np.min(x)
    y = y - np.min(y)

    x = x / (np.max(x) + 1e-9)
    y = y / (np.max(y) + 1e-9)

    xi = np.linspace(0, 1, 200)

    yi = []
    for v in xi:
        yi.append(lagrange(x, y, v))

    plt.plot(xi, yi)


# ---------------- FIXED DECODER ----------------
def decode(json_data):

    plt.figure(figsize=(8, 3))  # IMPORTANT RESET

    # your JSON is a LIST
    if isinstance(json_data, list):

        for char in json_data:
            if "points" in char:
                plot_points(char["points"])

    # sometimes Streamlit gives dict
    elif isinstance(json_data, dict):

        if "characters" in json_data:
            for char in json_data["characters"]:
                plot_points(char["points"])

    else:
        raise ValueError("Invalid JSON format")

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title("Decoded Lagrange Reconstruction")