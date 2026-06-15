import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from tensorflow.keras.models import load_model

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def explain_model_shap(
    data_path,
    model_path,
    scaler_path,
    output_dir="data/processed/shap",
    window_size=60,
    sample_size=50
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # ✅ FIX: LSTM sirf Close pe trained hai — Close hi use karo
    close_scaler = joblib.load(scaler_path)
    close_data   = df[["Close"]].values
    scaled_data  = close_scaler.fit_transform(close_data)

    # Sequences
    X, y   = create_sequences(scaled_data, window_size)
    split  = int(0.8 * len(X))
    X_test = X[split:]
    X_sample = X_test[:sample_size]  # shape: (50, 60, 1)

    # Load model
    model = load_model(model_path)

    print("⏳ SHAP values calculate ho rahe hain...")
    background   = X_sample[:20]    # shape: (20, 60, 1)
    explainer    = shap.GradientExplainer(model, background)
    shap_values  = explainer.shap_values(X_sample)  # shape: (50, 60, 1)
    shap_vals    = np.array(shap_values).squeeze()   # shape: (50, 60)

    # ─────────────────────────────────────────
    # Mean SHAP per time step
    # ─────────────────────────────────────────
    mean_shap  = np.mean(np.abs(shap_vals), axis=0)  # shape: (60,)

    # ✅ FIX: "Day -1" ki jagah meaningful labels
    # Last 10 days ko label karo, baaki ko group karo
    time_labels = []
    for i in range(window_size):
        days_ago = window_size - i
        if days_ago == 1:
            time_labels.append("Yesterday")
        elif days_ago <= 5:
            time_labels.append(f"{days_ago} days ago")
        elif days_ago <= 10:
            time_labels.append(f"{days_ago} days ago")
        elif days_ago <= 20:
            time_labels.append(f"~{days_ago}d ago\n(MA20 zone)")
        elif days_ago <= 50:
            time_labels.append(f"~{days_ago}d ago\n(MA50 zone)")
        else:
            time_labels.append(f"{days_ago}d ago")

    # ─────────────────────────────────────────
    # PLOT 1: Top 15 Most Influential Days
    # ─────────────────────────────────────────
    top15_idx    = np.argsort(mean_shap)[-15:]
    top15_shap   = mean_shap[top15_idx]
    top15_labels = [time_labels[i] for i in top15_idx]

    colors = []
    for v in top15_shap:
        if v == top15_shap.max():
            colors.append("#2E75B6")
        elif v >= np.percentile(top15_shap, 70):
            colors.append("#70AD47")
        else:
            colors.append("#ED7D31")

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(top15_labels, top15_shap, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, top15_shap):
        ax.text(bar.get_width() + 0.0001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Mean |SHAP Value|  —  Higher = more influence on prediction", fontsize=10)
    ax.set_title("Which Past Days Influenced Tomorrow's Price Prediction?",
                 fontsize=13, fontweight="bold", pad=15)

    legend_patches = [
        mpatches.Patch(color="#2E75B6", label="Most Influential Day"),
        mpatches.Patch(color="#70AD47", label="Moderately Influential"),
        mpatches.Patch(color="#ED7D31", label="Less Influential"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.set_xlim(0, top15_shap.max() * 1.25)
    plt.tight_layout()

    plot1_path = os.path.join(output_dir, "shap_feature_importance.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot 1 saved: {plot1_path}")

    # ─────────────────────────────────────────
    # PLOT 2: SHAP Over Time (line chart)
    # Shows overall pattern of influence
    # ─────────────────────────────────────────
    days_ago_axis = list(range(window_size, 0, -1))

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(days_ago_axis, mean_shap[::-1],
             color="#2E75B6", linewidth=2, marker="o", markersize=3)
    ax2.fill_between(days_ago_axis, mean_shap[::-1],
                     alpha=0.2, color="#2E75B6")

    # Highlight most influential day
    peak_idx     = np.argmax(mean_shap)
    peak_day_ago = window_size - peak_idx
    ax2.axvline(x=peak_day_ago, color="#C00000",
                linestyle="--", linewidth=1.5,
                label=f"Most influential: {peak_day_ago} days ago")

    ax2.set_xlabel("Days Before Today  (1 = Yesterday, 60 = 60 days ago)", fontsize=10)
    ax2.set_ylabel("Mean |SHAP Value|", fontsize=10)
    ax2.set_title("Influence of Past Days on Today's Prediction",
                  fontsize=13, fontweight="bold", pad=15)
    ax2.invert_xaxis()
    ax2.legend(fontsize=9)
    plt.tight_layout()

    plot2_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot 2 saved: {plot2_path}")

    # Top 3 most influential days
    top3_idx  = np.argsort(mean_shap)[-3:][::-1]
    top3_days = [f"{window_size - i} days ago" for i in top3_idx]
    top3_vals = [round(float(mean_shap[i]), 6) for i in top3_idx]

    print(f"\n🏆 Top 3 Most Influential Days: {top3_days}")

    return {
        "top3_days": top3_days,
        "top3_shap_values": top3_vals,
        "plot_feature_importance": plot1_path,
        "plot_summary": plot2_path
    }