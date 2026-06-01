import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

# ============================================================================
# CONFIGURATION: Set paths to your JSON files here
# ============================================================================
JSON_PATHS = {
    "en-vi": r"data\en-vi\eval\full_metrics\20260212_202624\scores.json",
    "en-tr": r"data\en-tr\eval\full_metrics\20260212_204246\scores.json",
    "en-th": r"data\en-th\eval\full_metrics\20260212_203936\scores.json",
    "en-sw": r"data\en-sw\eval\full_metrics\20260214_002351\scores.json",
}

# ============================================================================
# Load data from JSON files
# ============================================================================
data = {}
for lang_pair, json_path in JSON_PATHS.items():
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # Extract systems data and round COMET scores to 4 decimal places
    data[lang_pair] = {}
    for system, metrics in json_data["systems"].items():
        data[lang_pair][system] = {
            "comet": round(metrics["comet"], 4),
            "bleu": metrics["bleu"],
            "chrf": metrics["chrf"],
            "ter": metrics["ter"],
        }

print(f"Loaded data for {len(data)} language pairs")
print(f"Systems found: {list(data[list(data.keys())[0]].keys())}")

# ============================================================================
# Set professional HIGH-QUALITY style
# ============================================================================
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")

# Enhanced font settings for publication quality
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14

# High-quality rendering settings
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["savefig.format"] = "png"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1
plt.rcParams["figure.dpi"] = 100  # Screen display DPI
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["patch.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0

# Anti-aliasing and smoothing
plt.rcParams["text.antialiased"] = True
plt.rcParams["patch.antialiased"] = True
plt.rcParams["lines.antialiased"] = True

# System labels for display
system_labels = {
    "base": "Base (This Work)",
    "chatgpt-4.1": "ChatGPT-4.1",
    "google_translate": "Google Translate",
    "nllb": "NLLB",
    "nllb-600m": "NLLB-600M",
    "nllb-3.3b": "NLLB-3.3B",
}

# Language labels
lang_labels = {
    "en-vi": "English → Vietnamese",
    "en-tr": "English → Turkish",
    "en-th": "English → Thai",
    "en-sw": "English → Swahili",
}

# Colors - Professional scientific (Highlights 'base', muted colors for others)
colors = {
    "base": "#B22222",  # Firebrick Red (Strong highlight for 'This Work')
    "chatgpt-4.1": "#4682B4",  # Steel Blue
    "google_translate": "#708090",  # Slate Gray
    "nllb": "#99A3A4",  # Muted Gray
    "nllb-600m": "#B2BABB",  # Lighter Gray
    "nllb-3.3b": "#7F8C8D",  # Darker Gray
}

# ============================================================================
# Detect which systems are present in the data
# ============================================================================
all_systems = set()
for lang_pair in data:
    all_systems.update(data[lang_pair].keys())

# Define system order (base always first, then others)
system_order = ["base", "chatgpt-4.1", "google_translate"]
nllb_variants = sorted([s for s in all_systems if s.startswith("nllb")])
system_order.extend(nllb_variants)

# Filter to only systems that exist
systems_to_plot = [s for s in system_order if s in all_systems]

print(f"Systems to plot: {systems_to_plot}")

# ============================================================================
# Figure 1: COMET Score Comparison Across All Language Pairs
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

languages = list(data.keys())
x = np.arange(len(languages))
n_systems = len(systems_to_plot)
width = 0.8 / n_systems  # Distribute width evenly

# Plot each system with proper spacing
for i, system in enumerate(systems_to_plot):
    comet_scores = [data[lang][system]["comet"] for lang in languages]
    offset = (i - (n_systems - 1) / 2) * width
    bars = ax.bar(
        x + offset,
        comet_scores,
        width,
        label=system_labels.get(system, system),
        color=colors.get(system, "#999999"),
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )

    # Add value labels on bars with better formatting
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

ax.set_xlabel("Language Pair", fontweight="bold", fontsize=13)
ax.set_ylabel("COMET Score", fontweight="bold", fontsize=13)
ax.set_title(
    "COMET Score Comparison Across Translation Systems",
    fontweight="bold",
    pad=20,
    fontsize=15,
)
ax.set_xticks(x)
ax.set_xticklabels([lang_labels[lang] for lang in languages], fontsize=11)
ax.legend(loc="lower left", framealpha=0.95, edgecolor="black", fancybox=False)
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Auto-adjust y-axis to zoom in on differences
all_comet_scores = []
for lang in languages:
    for system in systems_to_plot:
        all_comet_scores.append(data[lang][system]["comet"])
min_comet = min(all_comet_scores)
max_comet = max(all_comet_scores)
padding = (max_comet - min_comet) * 0.1
ax.set_ylim(bottom=max(0, min_comet - padding), top=min(1.0, max_comet + padding))

plt.tight_layout()
plt.savefig(
    "viz/fig1_comet_comparison.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.savefig(
    "viz/fig1_comet_comparison.pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)  # Also save as PDF
plt.close()

print("✓ Figure 1: COMET Score Comparison created (PNG 600 DPI + PDF)")

# ============================================================================
# Figure 2: Percentage Improvement Over Baselines (COMET ONLY)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Calculate improvements for COMET only
improvements_data = []
for lang in languages:
    base_comet = data[lang]["base"]["comet"]
    for system in data[lang]:
        if system != "base":
            system_comet = data[lang][system]["comet"]
            improvement = ((base_comet - system_comet) / system_comet) * 100
            improvements_data.append(
                {
                    "Language": lang_labels[lang],
                    "System": system_labels.get(system, system),
                    "Improvement": improvement,
                }
            )

df_improvements = pd.DataFrame(improvements_data)

# Get competitor systems (everything except base)
competitor_systems = [system_labels.get(s, s) for s in systems_to_plot if s != "base"]

x = np.arange(len(languages))
n_competitors = len(competitor_systems)
width = 0.8 / n_competitors

for i, system in enumerate(competitor_systems):
    improvements = []
    for lang in languages:
        system_data = df_improvements[
            (df_improvements["System"] == system)
            & (df_improvements["Language"] == lang_labels[lang])
        ]
        if len(system_data) > 0:
            improvements.append(system_data["Improvement"].values[0])
        else:
            improvements.append(0)
    offset = (i - (n_competitors - 1) / 2) * width

    # Determine color
    color = None
    for sys_key, sys_label in system_labels.items():
        if sys_label == system:
            color = colors.get(sys_key, "#93c5fd")
            break
    if color is None:
        color = "#93c5fd"

    bars = ax.bar(
        x + offset,
        improvements,
        width,
        label=system,
        color=color,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )

    # Add value labels with better formatting
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

ax.set_xlabel("Language Pair", fontweight="bold", fontsize=13)
ax.set_ylabel("COMET Improvement (%)", fontweight="bold", fontsize=13)
ax.set_title(
    "Percentage Improvement of Base Model Over Existing Systems (COMET)",
    fontweight="bold",
    pad=20,
    fontsize=15,
)
ax.set_xticks(x)
ax.set_xticklabels([lang_labels[lang] for lang in languages], fontsize=11)
ax.legend(loc="upper left", framealpha=0.95, edgecolor="black", fancybox=False)
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.axhline(y=0, color="black", linestyle="-", linewidth=1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "viz/fig2_percentage_improvements.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.savefig(
    "viz/fig2_percentage_improvements.pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.close()

print("✓ Figure 2: Percentage Improvements (COMET) created (PNG 600 DPI + PDF)")

# ============================================================================
# Figure 3: All Metrics Comparison - Side by Side (NOT STARTING AT 0)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics_to_plot = ["comet", "bleu", "chrf", "ter"]
metric_titles = [
    "COMET Score",
    "BLEU Score",
    "chrF Score",
    "TER Score (Lower is Better)",
]

for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
    ax = axes[idx // 2, idx % 2]

    x = np.arange(len(languages))
    width = 0.8 / n_systems

    for i, system in enumerate(systems_to_plot):
        values = [data[lang][system][metric] for lang in languages]
        offset = (i - (n_systems - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=system_labels.get(system, system),
            color=colors.get(system, "#93c5fd"),
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )

        # Add value labels with better formatting
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("Language Pair", fontweight="bold", fontsize=12)
    ax.set_ylabel(metric.upper(), fontweight="bold", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [lang_labels[lang] for lang in languages], rotation=15, ha="right", fontsize=10
    )
    ax.legend(
        loc="best", framealpha=0.95, fontsize=9, edgecolor="black", fancybox=False
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set y-axis limits to zoom in on differences (not starting at 0)
    all_values = []
    for lang in languages:
        for system in systems_to_plot:
            all_values.append(data[lang][system][metric])

    min_val = min(all_values)
    max_val = max(all_values)
    padding = (max_val - min_val) * 0.15
    ax.set_ylim([min_val - padding, max_val + padding])

plt.suptitle(
    "Comprehensive Metric Comparison Across All Language Pairs",
    fontweight="bold",
    fontsize=16,
)
plt.tight_layout()
plt.savefig(
    "viz/fig3_all_metrics_comparison.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.savefig(
    "viz/fig3_all_metrics_comparison.pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.close()

print("✓ Figure 3: All Metrics Comparison created (PNG 600 DPI + PDF)")

print("\n" + "=" * 60)
print("ALL HIGH-QUALITY VISUALIZATIONS CREATED SUCCESSFULLY!")
print("=" * 60)
print("\nEnhancements applied:")
print("  • DPI increased to 600 (publication quality)")
print("  • Both PNG and PDF formats saved")
print("  • Enhanced fonts and sizes")
print("  • Better anti-aliasing")
print("  • Improved line widths and spacing")
print("  • Cleaner axes (removed top/right spines)")
print("  • Better legend styling")
print("  • White background for clean printing")
print("=" * 60)
