"""
ISO-NE LMP Price Visualizations
================================
Generates publication-quality figures for thesis analysis of
Real-Time and Day-Ahead hourly energy prices.

Usage:
    Place your CSVs in the same directory and update the file paths below.
    Run: python lmp_visualizations.py
    Figures are saved as high-resolution PNGs.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────


RT_FILE = "./data/isone_real_time_final_lmp_data.csv"

DA_FILE = "./data/isone_day_ahead_lmp_data.csv"

# Locations to highlight in multi-location plots.
# Pick 3–4 that are meaningful to your study area.
KEY_LOCATIONS = [
    "DR.CT_Eastern",
    "DR.CT_Northern",
    "DR.CT_Norwalk-Stamford",
    "DR.CT_Western",
]

# Rolling window for smoothing (hours)
ROLL_WINDOW = 168  # 7 days

DPI = 200
FORMAT = "png"

# ── Palette ───────────────────────────────────────────────────────────────────
# Soft, muted academic palette
BG = "#FAFAF8"
PANEL = "#F3F2EE"
GRID = "#E5E3DC"
TEXT = "#2C2C2C"
SUBTEXT = "#7A7A72"

# Muted accent colours per location / market
RT_COLOR = "#4A7FA5"   # muted steel blue  → Real-Time
DA_COLOR = "#C07A55"   # muted terracotta  → Day-Ahead
SPREAD_COLOR = "#6B9E78"  # muted sage green → Spread

LOC_COLORS = ["#4A7FA5", "#C07A55", "#6B9E78",
              "#9B7EBD"]  # blue, terra, sage, lavender

# ── Typography ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.facecolor":    BG,
    "figure.facecolor":  BG,
    "axes.edgecolor":    GRID,
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "grid.linestyle":    "-",
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "axes.labelcolor":   TEXT,
    "axes.titlesize":    12,
    "axes.titlecolor":   TEXT,
    "axes.titlepad":     12,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  GRID,
    "legend.fontsize":   9,
    "figure.dpi":        DPI,
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})

# ── Helpers ───────────────────────────────────────────────────────────────────


def load(path, label):
    df = pd.read_csv(path, parse_dates=["interval_start_utc"])
    df = df.rename(columns={"interval_start_utc": "ts", "lmp": "lmp"})
    df["ts"] = df["ts"].dt.tz_localize(None)   # strip tz for cleaner x-axis
    df["market"] = label
    return df


def avg_price(df):
    """Average LMP across all locations per timestamp."""
    return df.groupby("ts")["lmp"].mean().rename("lmp")


def key_locs(df):
    """Filter to KEY_LOCATIONS that actually exist in the data."""
    present = [l for l in KEY_LOCATIONS if l in df["location"].unique()]
    return df[df["location"].isin(present)]


def caption(ax, text):
    ax.text(0.0, -0.12, text, transform=ax.transAxes,
            fontsize=7.5, color=SUBTEXT, va="top")


def save(fig, name):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, f"{name}.{FORMAT}")
    fig.savefig(filepath)
    print(f"  Saved → {filepath}")


# ── Load & Prepare ────────────────────────────────────────────────────────────

print("Loading data …")
rt_raw = load(RT_FILE, "RT")
da_raw = load(DA_FILE, "DA")

rt = avg_price(rt_raw)
da = avg_price(da_raw)

combined = pd.DataFrame({"RT": rt, "DA": da}).dropna()
combined["spread"] = combined["RT"] - combined["DA"]
combined["month"] = combined.index.month
combined["hour"] = combined.index.hour

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Full-Year Price Time Series (RT & DA)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 1: Time Series …")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(combined.index, combined["RT"], lw=0.5,
        color=RT_COLOR, alpha=0.35, label="_nolegend_")
ax.plot(combined.index, combined["DA"], lw=0.5,
        color=DA_COLOR, alpha=0.35, label="_nolegend_")

rt_roll = combined["RT"].rolling(ROLL_WINDOW, center=True).mean()
da_roll = combined["DA"].rolling(ROLL_WINDOW, center=True).mean()

ax.plot(combined.index, rt_roll, lw=1.8,
        color=RT_COLOR, label="Real-Time (7-day avg)")
ax.plot(combined.index, da_roll, lw=1.8,
        color=DA_COLOR, label="Day-Ahead (7-day avg)")

ax.set_title("ISO-NE Hourly LMP — Real-Time vs. Day-Ahead, 2025")
ax.set_ylabel("LMP ($/MWh)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend(loc="upper right")
caption(ax, "Faint lines: raw hourly prices. Bold lines: 7-day centred rolling average. Averaged across all locations.")

fig.tight_layout()
save(fig, "fig1_timeseries")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — RT–DA Spread Time Series + Distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 2: Spread …")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4),
                               gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})

# Left: spread over time
ax1.axhline(0, color=SUBTEXT, lw=0.8, linestyle="--", alpha=0.6)
ax1.fill_between(combined.index, combined["spread"], 0,
                 where=combined["spread"] >= 0,
                 color=RT_COLOR, alpha=0.25, linewidth=0)
ax1.fill_between(combined.index, combined["spread"], 0,
                 where=combined["spread"] < 0,
                 color=DA_COLOR, alpha=0.25, linewidth=0)
ax1.plot(combined.index,
         combined["spread"].rolling(ROLL_WINDOW, center=True).mean(),
         lw=1.6, color=SPREAD_COLOR, label="7-day rolling mean")

ax1.set_title("RT–DA Price Spread (RT minus DA)")
ax1.set_ylabel("Spread ($/MWh)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.legend(loc="upper right")

# Right: distribution
spread_vals = combined["spread"].dropna()
bins = np.linspace(spread_vals.quantile(
    0.005), spread_vals.quantile(0.995), 60)
ax2.hist(spread_vals, bins=bins, orientation="horizontal",
         color=SPREAD_COLOR, alpha=0.7, edgecolor=BG, linewidth=0.4)
ax2.axhline(0, color=SUBTEXT, lw=0.8, linestyle="--", alpha=0.6)
ax2.axhline(spread_vals.mean(), color=SPREAD_COLOR, lw=1.4, linestyle=":")
ax2.set_title("Distribution")
ax2.set_xlabel("Hours")
ax2.yaxis.set_ticklabels([])
ax2.yaxis.set_ticks_position("none")
ax2.set_ylim(ax1.get_ylim() if ax1.get_ylim()[
             0] != ax1.get_ylim()[1] else None)

# Sync y-axes
y_lo = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_hi = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_lo, y_hi)
ax2.set_ylim(y_lo, y_hi)

caption(ax1, "Blue fill: RT > DA (arbitrage opportunity). Orange fill: RT < DA.")
fig.tight_layout()
save(fig, "fig2_spread")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Average Daily Price Profile by Hour-of-Day
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 3: Daily Profile …")

hourly_rt = combined.groupby("hour")["RT"].agg(["mean", "std"])
hourly_da = combined.groupby("hour")["DA"].agg(["mean", "std"])

fig, ax = plt.subplots(figsize=(9, 4))
hours = np.arange(24)

ax.fill_between(hours,
                hourly_rt["mean"] - hourly_rt["std"],
                hourly_rt["mean"] + hourly_rt["std"],
                color=RT_COLOR, alpha=0.12)
ax.fill_between(hours,
                hourly_da["mean"] - hourly_da["std"],
                hourly_da["mean"] + hourly_da["std"],
                color=DA_COLOR, alpha=0.12)

ax.plot(hours, hourly_rt["mean"], lw=2, color=RT_COLOR, marker="o",
        markersize=4, label="Real-Time")
ax.plot(hours, hourly_da["mean"], lw=2, color=DA_COLOR, marker="o",
        markersize=4, label="Day-Ahead")

ax.set_title("Average Hourly Price Profile — Full Year 2025")
ax.set_ylabel("Mean LMP ($/MWh)")
ax.set_xlabel("Hour of Day (UTC)")
ax.set_xticks(hours)
ax.set_xlim(-0.5, 23.5)
ax.legend()
caption(ax, "Shaded bands: ±1 standard deviation. UTC hours shown.")

fig.tight_layout()
save(fig, "fig3_daily_profile")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Price Duration Curve
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 4: Price Duration Curve …")

fig, ax = plt.subplots(figsize=(9, 4))

for col, color, label in [("RT", RT_COLOR, "Real-Time"),
                          ("DA", DA_COLOR, "Day-Ahead")]:
    vals = combined[col].dropna().sort_values(ascending=False).values
    pct = np.linspace(0, 100, len(vals))
    ax.plot(pct, vals, lw=1.8, color=color, label=label)

ax.axhline(0, color=SUBTEXT, lw=0.7, linestyle="--", alpha=0.5)
ax.set_title("Price Duration Curve — 2025")
ax.set_xlabel("Percentage of Hours (%)")
ax.set_ylabel("LMP ($/MWh)")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend()
caption(ax, "Prices sorted descending. Shows proportion of hours above any given price threshold.")

fig.tight_layout()
save(fig, "fig4_duration_curve")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Price Heatmap (Hour × Month)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 5: Heatmap …")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

for col, color_base, label, fname in [
    ("RT", RT_COLOR,  "Real-Time", "fig5a_heatmap_rt"),
    ("DA", DA_COLOR,  "Day-Ahead", "fig5b_heatmap_da"),
]:
    pivot = combined.pivot_table(
        values=col, index="hour", columns="month", aggfunc="mean")

    # Build a soft colormap from near-white → accent colour
    cmap = LinearSegmentedColormap.from_list(
        "soft", ["#F7F4EF", color_base], N=256
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot, aspect="auto", cmap=cmap, interpolation="bilinear")

    ax.set_title(f"Mean {label} LMP by Hour and Month — 2025 ($/MWh)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of Day (UTC)")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([MONTH_LABELS[m-1] for m in pivot.columns])
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("$/MWh", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    caption(ax, "Bilinear interpolation applied for visual smoothness.")
    fig.tight_layout()
    save(fig, fname)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Monthly Average Price (RT & DA)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 6: Monthly Averages …")

monthly = combined.groupby("month")[["RT", "DA"]].mean()
x = np.arange(1, 13)
w = 0.35

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - w/2, monthly["RT"], width=w,
       color=RT_COLOR, alpha=0.85, label="Real-Time")
ax.bar(x + w/2, monthly["DA"], width=w,
       color=DA_COLOR, alpha=0.85, label="Day-Ahead")

ax.set_title("Monthly Average LMP — 2025")
ax.set_ylabel("Mean LMP ($/MWh)")
ax.set_xlabel("Month")
ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.legend()
caption(ax, "Averaged across all hours and locations within each month.")

fig.tight_layout()
save(fig, "fig6_monthly_avg")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Key Locations: Annual Average RT Price Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 7: Key Locations …")

rt_locs = key_locs(rt_raw)
hourly_by_loc = (rt_locs
                 .groupby(["location", rt_locs["ts"].dt.hour])["lmp"]
                 .mean()
                 .unstack(level=0))

fig, ax = plt.subplots(figsize=(10, 4))

for i, loc in enumerate(hourly_by_loc.columns):
    short = loc.replace("DR.CT_", "")  # cleaner label
    ax.plot(hourly_by_loc.index, hourly_by_loc[loc],
            lw=1.8, color=LOC_COLORS[i % len(LOC_COLORS)],
            marker="o", markersize=3.5, label=short)

ax.set_title("Average Hourly Real-Time LMP by Location — 2025")
ax.set_ylabel("Mean LMP ($/MWh)")
ax.set_xlabel("Hour of Day (UTC)")
ax.set_xticks(range(24))
ax.set_xlim(-0.5, 23.5)
ax.legend(title="Location", title_fontsize=8)
caption(ax, "Annual hourly average per location. Useful for identifying locational congestion patterns.")

fig.tight_layout()
save(fig, "fig7_locations")
plt.close()

print("\nAll figures saved successfully.")
