"""
=============================================================================
Linear Regression Simulation — Interactive Control Bar
=============================================================================
Controls: a (slope), b (intercept), n (points), var (variance)
Sliders update the plot in real-time.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

RANDOM_SEED = 42
X_MIN, X_MAX = -100, 100
N_OUTLIERS = 10

# --- Initial defaults ---
N_INIT = 200
A_INIT = 2.5
B_INIT = 10.0
VAR_INIT = 300.0

# --- Generate initial data ---
rng = np.random.default_rng(RANDOM_SEED)
x = rng.uniform(X_MIN, X_MAX, size=N_INIT)
noise = rng.normal(loc=0.0, scale=np.sqrt(VAR_INIT), size=N_INIT)
y = A_INIT * x + B_INIT + noise

# --- Initial model ---
model = LinearRegression()
X = x.reshape(-1, 1)
model.fit(X, y)
a_pred = model.coef_[0]
b_pred = model.intercept_
y_pred_all = model.predict(X)
residuals = np.abs(y - y_pred_all)
outlier_indices = np.argsort(residuals)[::-1][:N_OUTLIERS]
r2 = r2_score(y, y_pred_all)

# =============================================================================
# Setup Figure
# =============================================================================
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#0f0f1a")

# Plot area
ax = fig.add_axes([0.08, 0.30, 0.90, 0.65])
ax.set_facecolor("#16213e")

# --- Slider axes ---
slider_left = 0.12
slider_width = 0.30
slider_height = 0.025
slider_y_positions = [0.20, 0.15, 0.10, 0.05]
slider_labels_positions = [0.225, 0.175, 0.125, 0.075]

ax_slider_a = fig.add_axes([slider_left, slider_y_positions[0], slider_width, slider_height])
ax_slider_b = fig.add_axes([slider_left, slider_y_positions[1], slider_width, slider_height])
ax_slider_n = fig.add_axes([slider_left + slider_width + 0.20, slider_y_positions[0], slider_width, slider_height])
ax_slider_var = fig.add_axes([slider_left + slider_width + 0.20, slider_y_positions[1], slider_width, slider_height])

# --- Sliders ---
slider_a = Slider(ax_slider_a, "slope (a)", -50, 50, valinit=A_INIT, valfmt="%.1f", color="#4fc3f7")
slider_b = Slider(ax_slider_b, "intercept (b)", -200, 200, valinit=B_INIT, valfmt="%.1f", color="#4fc3f7")
slider_n = Slider(ax_slider_n, "points (n)", 10, 1000, valinit=N_INIT, valfmt="%d", color="#ffd700", valstep=1)
slider_var = Slider(ax_slider_var, "variance (var)", 1, 1000, valinit=VAR_INIT, valfmt="%.0f", color="#ffd700")

for s in [slider_a, slider_b, slider_n, slider_var]:
    s.label.set_color("white")
    s.label.set_fontsize(11)
    s.valtext.set_color("#aaaacc")

# --- Reset button ---
ax_reset = fig.add_axes([0.81, 0.08, 0.08, 0.05])
btn_reset = Button(ax_reset, "Reset", color="#1e1e3a", hovercolor="#3a3a6a")
btn_reset.label.set_color("white")

# =============================================================================
# Plotting function
# =============================================================================
(scatter_all,) = ax.plot([], [], "o", color="#4fc3f7", alpha=0.55, markersize=5, zorder=2)
(scatter_outliers,) = ax.plot([], [], "o", markersize=12, markerfacecolor="none",
                                markeredgecolor="#ffd700", markeredgewidth=2.2, zorder=4)
(line_reg,) = ax.plot([], [], "-", color="#ff4757", linewidth=2.5, zorder=3)
residual_lines = []

ax.set_xlim(X_MIN - 10, X_MAX + 10)
ax.tick_params(colors="#aaaacc")
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
ax.set_xlabel("x", color="#aaaacc", fontsize=13)
ax.set_ylabel("y", color="#aaaacc", fontsize=13)

def update(val=None):
    global residual_lines

    a_true = slider_a.val
    b_true = slider_b.val
    n = int(slider_n.val)
    var_true = slider_var.val

    rng_local = np.random.default_rng(RANDOM_SEED)
    x_new = rng_local.uniform(X_MIN, X_MAX, size=n)
    noise_new = rng_local.normal(loc=0.0, scale=np.sqrt(var_true), size=n)
    y_new = a_true * x_new + b_true + noise_new

    X_new = x_new.reshape(-1, 1)
    model_local = LinearRegression()
    model_local.fit(X_new, y_new)
    a_pred_new = model_local.coef_[0]
    b_pred_new = model_local.intercept_
    y_pred_new = model_local.predict(X_new)
    residuals_new = np.abs(y_new - y_pred_new)
    outlier_idx_new = np.argsort(residuals_new)[::-1][:N_OUTLIERS]
    r2_new = r2_score(y_new, y_pred_new)

    # Update scatter
    scatter_all.set_data(x_new, y_new)

    # Update regression line
    x_line = np.linspace(X_MIN, X_MAX, 400)
    y_line = a_pred_new * x_line + b_pred_new
    line_reg.set_data(x_line, y_line)

    # Update outliers
    x_out = x_new[outlier_idx_new]
    y_out = y_new[outlier_idx_new]
    scatter_outliers.set_data(x_out, y_out)

    # Update residual drop lines
    for li in residual_lines:
        li.remove()
    residual_lines.clear()
    for xi, yi in zip(x_out, y_out):
        yi_hat = a_pred_new * xi + b_pred_new
        (li,) = ax.plot(
            [xi, xi], [yi, yi_hat],
            color="#ffd700", linewidth=1, linestyle="--", alpha=0.6, zorder=3
        )
        residual_lines.append(li)

    # Rescale y-axis
    margin = (y_new.max() - y_new.min()) * 0.15
    ax.set_ylim(y_new.min() - margin, y_new.max() + margin)

    # Update title
    true_eq = f"True: y = {a_true:.2f}x + {b_true:.2f}  |  var = {var_true:.0f}"
    pred_eq = f"Predicted: y^ = {a_pred_new:.2f}x + {b_pred_new:.2f}  |  R^2 = {r2_new:.4f}"
    ax.set_title(f"{true_eq}\n{pred_eq}", color="white", fontsize=12, pad=14, fontfamily="monospace")

    fig.canvas.draw_idle()

def reset(event):
    slider_a.reset()
    slider_b.reset()
    slider_n.reset()
    slider_var.reset()

slider_a.on_changed(update)
slider_b.on_changed(update)
slider_n.on_changed(update)
slider_var.on_changed(update)
btn_reset.on_clicked(reset)

update()

plt.show()
