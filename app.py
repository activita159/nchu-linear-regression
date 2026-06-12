import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

RANDOM_SEED = 42
X_MIN, X_MAX = -100, 100
N_OUTLIERS = 10

st.set_page_config(page_title="Linear Regression Simulation", layout="wide")
st.title("Linear Regression Simulation")

col1, col2 = st.columns(2)
with col1:
    a_true = st.slider("Slope (a)", -50.0, 50.0, 2.5, 0.1)
    b_true = st.slider("Intercept (b)", -200.0, 200.0, 10.0, 0.1)
with col2:
    n = st.slider("Points (n)", 10, 1000, 200, 1)
    var_true = st.slider("Variance (var)", 1, 1000, 300, 1)

rng = np.random.default_rng(RANDOM_SEED)
x = rng.uniform(X_MIN, X_MAX, size=n)
noise = rng.normal(loc=0.0, scale=np.sqrt(var_true), size=n)
y = a_true * x + b_true + noise

X = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
a_pred = model.coef_[0]
b_pred = model.intercept_
y_pred = model.predict(X)
residuals = np.abs(y - y_pred)
outlier_indices = np.argsort(residuals)[::-1][:N_OUTLIERS]
r2 = r2_score(y, y_pred)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#16213e")

ax.scatter(x, y, color="#4fc3f7", alpha=0.55, s=25, zorder=2, label="Data points")

x_out = x[outlier_indices]
y_out = y[outlier_indices]
ax.scatter(x_out, y_out, s=144, facecolors="none", edgecolors="#ffd700", linewidths=2.2, zorder=4, label="Top 10 outliers")

x_line = np.linspace(X_MIN, X_MAX, 400)
y_line = a_pred * x_line + b_pred
ax.plot(x_line, y_line, color="#ff4757", linewidth=2.5, zorder=3, label="Regression line")

for xi, yi in zip(x_out, y_out):
    yi_hat = a_pred * xi + b_pred
    ax.plot([xi, xi], [yi, yi_hat], color="#ffd700", linewidth=1, linestyle="--", alpha=0.6, zorder=3)

margin = (y.max() - y.min()) * 0.15
ax.set_ylim(y.min() - margin, y.max() + margin)
ax.set_xlim(X_MIN - 10, X_MAX + 10)

ax.tick_params(colors="#aaaacc")
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
ax.set_xlabel("x", color="#aaaacc", fontsize=13)
ax.set_ylabel("y", color="#aaaacc", fontsize=13)

true_eq = f"True: y = {a_true:.2f}x + {b_true:.2f}  |  var = {var_true:.0f}"
pred_eq = f"Predicted: y\u0302 = {a_pred:.2f}x + {b_pred:.2f}  |  R\u00b2 = {r2:.4f}"
ax.set_title(f"{true_eq}\n{pred_eq}", color="white", fontsize=12, pad=14, fontfamily="monospace")
ax.legend(facecolor="#1e1e3a", edgecolor="#333355", labelcolor="white")

st.pyplot(fig)
