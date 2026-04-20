import numpy as np
import matplotlib.pyplot as plt
import json

# ---------------- LAGRANGE ----------------
def lagrange(x, y, xi):
    total = 0
    n = len(x)

    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (xi - x[j]) / (x[i] - x[j])
        total += term

    return total


# ---------------- DRAW ----------------
def reconstruct_and_plot(key_data):

    plt.figure(figsize=(8,3))

    result_text = ""

    offset = 0

    for char in key_data:

        pts = np.array(char["points"])
        label = char.get("char", "?")

        x = pts[:,0]
        y = pts[:,1]

        xi_vals = np.linspace(min(x), max(x), 200)
        yi_vals = [lagrange(x, y, xi) for xi in xi_vals]

        # shift characters horizontally
        plt.plot(xi_vals + offset, yi_vals, linewidth=2)

        offset += max(x) - min(x) + 20

        result_text += label

    plt.gca().invert_yaxis()
    plt.title(result_text)

    plt.savefig("result.png")
    plt.close()

    return result_text