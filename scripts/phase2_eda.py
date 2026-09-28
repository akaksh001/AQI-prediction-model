"""
Phase 2 — Exploratory Data Analysis (EDA)
DSN2098 · Group 65 · Air Quality Analytics
Owner: Yugank Shakya

Reads:  data/processed/clean_aqi.csv   (produced by Phase 1)
Saves:  outputs/eda/
    01_correlation_heatmap.png
    02_aqi_kde_by_season.png
    03_aqi_timeseries.png
    04_class_distribution.png
    05_monthly_heatmap.png
    06_pollutant_vs_aqi.png
    07_geo_scatter.html

Usage:
    python scripts/phase2_eda.py
    (run from D:/projectexhibit-aiml/)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CLEAN_PATH   = "data/processed/clean_aqi.csv"
PLOT_DIR     = "outputs/eda"

POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx",
                  "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

# Delhi station coordinates (single station, Phase 3 adds multi-station)
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

# Indian seasonal mapping (month -> season)
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer",  4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
}

SEASON_ORDER  = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
SEASON_COLORS = {"Winter": "#5b9bd5", "Summer": "#ed7d31",
                 "Monsoon": "#70ad47", "Post-Monsoon": "#ffc000"}

BUCKET_ORDER  = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BUCKET_COLORS = ["#00b050", "#92d050", "#ffff00", "#ff7f00", "#ff0000", "#7030a0"]

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def save(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")

def check_file(path):
    if not os.path.exists(path):
        print(f"\nERROR: {path} not found. Run phase1_clean.py first.\n")
        sys.exit(1)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────
    section("Load clean_aqi.csv")
    check_file(CLEAN_PATH)
    df = pd.read_csv(CLEAN_PATH, parse_dates=["datetime"], index_col="datetime")
    print(f"  Shape  : {df.shape}")
    print(f"  Dates  : {df.index.min().date()} -> {df.index.max().date()}")

    # Add time features
    df["season"] = df.index.month.map(SEASON_MAP)
    df["month"]  = df.index.month
    df["year"]   = df.index.year
    df["hour"]   = df.index.hour

    # Fix: keep only pollutant cols that exist
    poll_cols = [c for c in POLLUTANT_COLS if c in df.columns]

    # ── 1. Correlation Heatmap ─────────────────
    section("Plot 1 — Correlation Heatmap")
    corr_cols  = poll_cols + ["AQI"]
    corr       = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))      # upper triangle masked
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap=cmap, center=0, linewidths=0.5,
                annot_kws={"size": 8}, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Pollutant Correlation Matrix — Delhi (2015-2020)",
                 fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    save(fig, "01_correlation_heatmap.png")

    # Top 5 correlators with AQI
    top5 = corr["AQI"].drop("AQI").abs().sort_values(ascending=False).head(5)
    print("\n  Top 5 pollutants correlated with AQI:")
    for col, val in top5.items():
        print(f"    {col:<12} r = {corr['AQI'][col]:+.3f}")

    # ── 2. AQI KDE by Season ──────────────────
    section("Plot 2 — AQI KDE by Season")
    fig, ax = plt.subplots(figsize=(10, 5))
    for season in SEASON_ORDER:
        subset = df.loc[df["season"] == season, "AQI"].dropna()
        color  = SEASON_COLORS[season]
        sns.kdeplot(subset, label=f"{season} (n={len(subset):,})",
                    fill=True, alpha=0.25, linewidth=2,
                    color=color, ax=ax)

    # CPCB bucket boundary lines
    for boundary, label in zip([50, 100, 200, 300, 400],
                                ["Good|Sat.", "Sat.|Mod.", "Mod.|Poor",
                                 "Poor|V.Poor", "V.Poor|Severe"]):
        ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(boundary + 2, ax.get_ylim()[1] * 0.95, label,
                fontsize=7, color="grey", va="top")

    ax.set_xlabel("AQI", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("AQI Distribution by Season — Delhi", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 510)
    save(fig, "02_aqi_kde_by_season.png")

    # ── 3. AQI Time-Series ────────────────────
    section("Plot 3 — AQI Time-Series")
    daily_aqi = df["AQI"].resample("D").mean()

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(daily_aqi.index, daily_aqi.values,
            color="#4472c4", linewidth=0.6, alpha=0.7)

    # 30-day rolling mean overlay
    rolling = daily_aqi.rolling(30, center=True).mean()
    ax.plot(rolling.index, rolling.values,
            color="#ed7d31", linewidth=2.0, label="30-day rolling mean")

    # CPCB zone shading
    zones = [(0, 50, "#00b050", 0.08),   (50, 100, "#92d050", 0.08),
             (100, 200, "#ffff00", 0.08), (200, 300, "#ff7f00", 0.08),
             (300, 400, "#ff0000", 0.08), (400, 500, "#7030a0", 0.08)]
    for lo, hi, col, alpha in zones:
        ax.axhspan(lo, hi, color=col, alpha=alpha)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Daily Average AQI", fontsize=10)
    ax.set_title("Delhi AQI Trend 2015-2020  (grey = raw daily, orange = 30-day mean)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 510)
    save(fig, "03_aqi_timeseries.png")

    # ── 4. Class Distribution Bar ─────────────
    section("Plot 4 — AQI Category Distribution")
    counts = df["AQI_Bucket"].value_counts().reindex(BUCKET_ORDER).dropna()
    pct    = (counts / counts.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=BUCKET_COLORS[:len(counts)], edgecolor="white", linewidth=0.8)
    for bar, p in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 80,
                f"{p}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("AQI Category", fontsize=11)
    ax.set_ylabel("Number of Hours", fontsize=11)
    ax.set_title("AQI Category Distribution — Delhi (Class Imbalance Check)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    save(fig, "04_class_distribution.png")
    print("\n  Category counts:")
    for cat, cnt, p in zip(counts.index, counts.values, pct.values):
        bar = "#" * int(p / 2)
        print(f"    {cat:<14} {cnt:>6,}  ({p:>5.1f}%)  {bar}")

    # ── 5. Monthly AQI Heatmap ────────────────
    section("Plot 5 — Monthly AQI Heatmap (Year x Month)")
    pivot = df.pivot_table(values="AQI", index="year",
                           columns="month", aggfunc="mean")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:pivot.shape[1]]

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=50, vmax=400, linewidths=0.5,
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Monthly Mean AQI — Delhi (Year vs Month)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Year")
    save(fig, "05_monthly_heatmap.png")

    # ── 6. Top Pollutants vs AQI Scatter ──────
    section("Plot 6 — Top Pollutants vs AQI")
    top_poll = top5.index.tolist()[:4]   # use top-4
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, col in zip(axes, top_poll):
        sample = df[[col, "AQI", "season"]].dropna().sample(
                    min(3000, len(df)), random_state=42)
        for season in SEASON_ORDER:
            s = sample[sample["season"] == season]
            ax.scatter(s[col], s["AQI"],
                       color=SEASON_COLORS[season], alpha=0.3,
                       s=8, label=season)
        # trend line
        z = np.polyfit(sample[col].values, sample["AQI"].values, 1)
        p = np.poly1d(z)
        xr = np.linspace(sample[col].min(), sample[col].max(), 100)
        ax.plot(xr, p(xr), color="black", linewidth=1.5, linestyle="--")
        r = sample[[col, "AQI"]].corr().iloc[0, 1]
        ax.set_xlabel(col, fontsize=10)
        ax.set_title(f"r = {r:+.3f}", fontsize=10)
        ax.set_ylabel("AQI" if ax == axes[0] else "", fontsize=10)

    handles = [plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=SEASON_COLORS[s], markersize=7, label=s)
               for s in SEASON_ORDER]
    fig.legend(handles=handles, loc="upper right",
               fontsize=9, title="Season", ncol=2)
    fig.suptitle("Top Pollutants vs AQI (colored by season)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "06_pollutant_vs_aqi.png")

    # ── 7. Geo Scatter (Plotly) ───────────────
    section("Plot 7 — Geospatial AQI Scatter")

    # Add coordinates (single station — slight jitter shows density by AQI bucket)
    rng = np.random.default_rng(42)
    sample_geo = df[["AQI", "AQI_Bucket"]].dropna().copy()
    sample_geo["lat"] = DELHI_LAT + rng.normal(0, 0.04, len(sample_geo))
    sample_geo["lon"] = DELHI_LON + rng.normal(0, 0.04, len(sample_geo))
    sample_geo = sample_geo.sample(min(5000, len(sample_geo)), random_state=42)

    color_map = dict(zip(BUCKET_ORDER, BUCKET_COLORS))
    sample_geo["AQI_Bucket"] = pd.Categorical(
        sample_geo["AQI_Bucket"], categories=BUCKET_ORDER, ordered=True)
    sample_geo = sample_geo.sort_values("AQI_Bucket")

    fig_geo = px.scatter_geo(
        sample_geo, lat="lat", lon="lon",
        color="AQI_Bucket",
        color_discrete_map=color_map,
        category_orders={"AQI_Bucket": BUCKET_ORDER},
        hover_data={"AQI": ":.1f", "lat": False, "lon": False},
        title="Delhi AQI — Geospatial Distribution (5,000 sample points)",
        projection="natural earth",
        opacity=0.6,
        size_max=6,
    )
    fig_geo.update_geos(
        center=dict(lat=DELHI_LAT, lon=DELHI_LON),
        lataxis_range=[27.5, 29.5],
        lonaxis_range=[76.0, 78.5],
        showland=True, landcolor="#f5f5f0",
        showocean=True, oceancolor="#cce5ff",
        showlakes=True, lakecolor="#cce5ff",
        showcountries=True, countrycolor="#aaaaaa",
    )
    fig_geo.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    geo_path = os.path.join(PLOT_DIR, "07_geo_scatter.html")
    fig_geo.write_html(geo_path)
    print(f"  Saved -> {geo_path}")

    # ── Summary ───────────────────────────────
    section("Phase 2 Complete — Summary")
    plots = [
        "01_correlation_heatmap.png",
        "02_aqi_kde_by_season.png",
        "03_aqi_timeseries.png",
        "04_class_distribution.png",
        "05_monthly_heatmap.png",
        "06_pollutant_vs_aqi.png",
        "07_geo_scatter.html",
    ]
    print(f"\n  All outputs saved to: {PLOT_DIR}/\n")
    for p in plots:
        full = os.path.join(PLOT_DIR, p)
        size = os.path.getsize(full)
        print(f"    {'OK':>3}  {p}  ({size/1024:.1f} KB)")
    print(f"\n  Phase 2 complete. Hand off clean_aqi.csv to Phase 3 (Akaksh).\n")


if __name__ == "__main__":
    main()
