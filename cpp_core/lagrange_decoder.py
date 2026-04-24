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


# ---------------- PLOT ----------------
def plot_points(pts):
    pts = np.array(pts)

    x = pts[:, 0]
    y = pts[:, 1]

    # normalize for stability
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
    y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-9)

    xi = np.linspace(0, 1, 200)
    yi = [lagrange(x, y, v) for v in xi]

    plt.plot(xi, yi)
    plt.gca().invert_yaxis()
    plt.axis("equal")


# ---------------- FIXED DECODER ----------------
def decode(json_data):

    # IMPORTANT FIX: json_data is LIST, not dict
    for char in json_data:
        pts = char["points"]
        plot_points(pts)

    plt.show()