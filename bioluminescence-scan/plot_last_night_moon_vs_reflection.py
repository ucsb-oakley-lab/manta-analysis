from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT = Path("/Users/oakley/temporary/output")
INPUT = OUTPUT / "bioluminescence_hits_classified.csv"
FIGURE = OUTPUT / "20261117_moon_vs_sand_reflection.png"

df = pd.read_csv(INPUT)
df = df[df["night"].astype(str) == "20261117"].copy()
df["time"] = pd.to_datetime(df["inferred_simulator_time"], errors="coerce")
df = df.dropna(subset=["time", "moon_illumination_fraction"])

reflection = df[df["classification"] == "reflection_suspect"].copy()
reflection["brightness"] = pd.to_numeric(reflection["reference_brightness"], errors="coerce")

bins = pd.date_range(df["time"].min().floor("5min"), df["time"].max().ceil("5min"), freq="5min")
counts = reflection.set_index("time").resample("5min").size().reindex(bins, fill_value=0)
illumination = df.set_index("time")["moon_illumination_fraction"].resample("5min").mean().reindex(bins).interpolate()

fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

ax = axes[0]
ax2 = ax.twinx()
ax.step(bins, counts.values, where="mid", color="firebrick", linewidth=2, label="sand-reflection detections / 5 min")
ax2.plot(bins, illumination.values, color="black", linewidth=2, label="predicted moon illumination")
ax.set_ylabel("Reflection detections")
ax2.set_ylabel("Moon illumination fraction")
ax.set_title("2026-11-17: predicted moon state vs. sand reflections")
ax.grid(True, alpha=0.25)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper left")

ax = axes[1]
reflection["moon_illumination_fraction"] = pd.to_numeric(
    reflection["moon_illumination_fraction"], errors="coerce"
)
reflection = reflection.dropna(subset=["brightness", "moon_illumination_fraction"])
ax.scatter(
    reflection["moon_illumination_fraction"],
    reflection["brightness"],
    s=12,
    alpha=0.5,
    color="firebrick",
    edgecolors="none",
)
ax.set_xlabel("Predicted moon illumination fraction")
ax.set_ylabel("Sand-reference brightness")
ax.set_title("Individual detections classified as sand reflection")
ax.grid(True, alpha=0.25)

fig.savefig(FIGURE, dpi=180)
print(f"Saved {FIGURE}")
print(f"Rows plotted: {len(df)}; sand reflections: {len(reflection)}")
