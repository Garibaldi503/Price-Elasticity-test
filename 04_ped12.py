# 04_price_elasticity_ped.py
# -----------------------------------------------------------
# Streamlit PED app — CLEAN build
# Features:
# - Built-in hardware demo data (no external CSV required) OR upload CSV
# - Log–log OLS per SKU: ln(Q) = a + b ln(P)
# - All-in-one chart: Q (left y), Revenue & GP (right y)
# - Shows current price, lets you CHOOSE price via SLIDER (live updates)
# - Quantity normalization toggle; Revenue/GP axis scaling (raw, thousands, millions)
# - Optimal price suggestion for GP (interior if b < -1), revenue boundary guidance
# - Target Margin % solver with price marking on chart
# - Consistent colors & legend:
#     Quantity = blue, Revenue = green, GP = gold
#     Current price = brown, User price = green, Opt GP = orange, Target Margin = purple
#
# Run:
#   pip install streamlit pandas numpy matplotlib
#   streamlit run 04_price_elasticity_ped.py
#
# CSV expected columns (case-insensitive):
#   product_id | date | price | quantity
# Optional: unit_cost (else assumed = 70% of median price per SKU)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="PED — All-in-One (Clean)", layout="wide")

NAVY = "#0a2342"
GOLD_HEX = "#c7a008"

st.markdown(
    f"""
    <style>
      :root {{ --navy: {NAVY}; --gold: {GOLD_HEX}; }}
      .title {{ color: var(--navy); font-weight: 800; font-size: 1.8rem; margin-bottom: 0.3rem; }}
      .sub {{ color: #444; font-size: 0.95rem; margin-bottom: 0.6rem; }}
      .kpi {{ border: 1px solid #eee; padding: 10px 12px; border-radius: 12px; background: #fff; }}
      .kpi h3 {{ margin: 0 0 6px 0; color: var(--navy); font-size: 1.0rem; }}
      .kpi p {{ margin: 0; font-size: 0.95rem; color: #333; }}
      .section {{ font-weight: 700; color: var(--navy); border-left: 6px solid var(--gold); padding-left: 8px; margin-top: 0.6rem;}}
      .stButton>button {{ background: var(--navy); color: white; border-radius: 999px; }}
      .stDownloadButton>button {{ background: var(--gold); color: black; border-radius: 999px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Price Elasticity of Demand (PED) — All-in-One</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Single chart view with live slider. Model: ln(Q) = a + b·ln(P).</div>', unsafe_allow_html=True)

# ---------------- HELPERS ----------------
ALIASES = {
    "product_id": ["product_id", "product", "sku", "item", "product code", "productcode"],
    "date": ["date", "dt", "order_date", "trans_date"],
    "price": ["price", "unit_price", "selling_price", "sellprice", "sales_price"],
    "quantity": ["quantity", "qty", "units", "sales_units", "sales_qty", "q"],
    "unit_cost": ["unit_cost", "cost", "cost_price", "cogs_unit", "cost_per_unit"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for std, alts in ALIASES.items():
        for a in alts:
            if a in cols:
                mapping[cols[a]] = std
                break
    df = df.rename(columns=mapping)
    req = {"product_id", "date", "price", "quantity"}
    have = set(df.columns.str.lower())
    if not req.issubset(have):
        missing = sorted(list(req - have))
        st.error(f"Missing required columns: {missing}. Need product_id, date, price, quantity.")
        st.stop()
    df.columns = [c.lower().strip() for c in df.columns]
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df

@dataclass
class FitResult:
    a: float
    b: float
    r2: float
    n: int

def fit_loglog_elasticity(df: pd.DataFrame) -> FitResult:
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["price", "quantity"])
    d = d[(d["price"] > 0) & (d["quantity"] > 0)]
    n = len(d)
    if n < 5:
        return FitResult(np.nan, np.nan, np.nan, n)
    x = np.log(d["price"].values)
    y = np.log(d["quantity"].values)
    b, a = np.polyfit(x, y, 1)  # slope b, intercept a
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return FitResult(a, b, r2, n)

def classify(b: float) -> str:
    if not np.isfinite(b):
        return "N/A"
    ab = abs(b)
    if ab > 1.05: return "Elastic"
    if ab < 0.95: return "Inelastic"
    return "Unitary (~1)"

def demand(a: float, b: float, p: np.ndarray) -> np.ndarray:
    return np.exp(a) * np.power(p, b)

def unit_cost_guess(price_series: pd.Series, unit_cost_series: Optional[pd.Series]) -> float:
    if unit_cost_series is not None and unit_cost_series.notna().any():
        return float(unit_cost_series.median())
    return float(price_series.median() * 0.70)

def money(x: float, cur: str="R") -> str:
    try: return f"{cur} {x:,.2f}"
    except: return f"{cur} {x}"

def opt_prices_for_range(a: float, b: float, c: float, lo: float, hi: float) -> dict:
    """Interior GP-optimal price exists when b < -1: P* = c b / (b+1).
       Revenue extremum is at boundary unless b ≈ -1 (flat)."""
    out = {}
    p_gp_interior = None
    if b < -1:
        p_star = (c * b) / (b + 1.0)
        if np.isfinite(p_star) and p_star > 0:
            p_gp_interior = p_star
    def clamp(x): 
        return min(max(x, lo), hi) if x is not None else None
    out["p_gp_interior"] = p_gp_interior
    out["p_gp_display"] = clamp(p_gp_interior) if p_gp_interior is not None else None

    if abs(b + 1.0) < 1e-6:
        out["p_rev_boundary"] = None
    elif b + 1.0 < 0:
        out["p_rev_boundary"] = lo
    else:
        out["p_rev_boundary"] = hi
    return out

# ---------------- DEMO DATA ----------------
@st.cache_data
def demo(rows_per_sku: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skus = [
        "HAMMER-1KG","DRILL-500W","SCREW-50MM","NAIL-100PC","PAINT-5L",
        "SANDPAPER-P80","WRENCH-SET","SAFETY-GLOVES","TAPE-MEASURE","LADDER-3M"
    ]
    start = pd.Timestamp("2024-01-01")
    weeks = pd.date_range(start, periods=rows_per_sku, freq="W")
    rows = []
    for sku in skus:
        base_price = rng.uniform(50, 1500)
        true_b = -rng.uniform(0.4, 1.8)
        a = rng.uniform(2.0, 5.0)
        for dt in weeks:
            price = base_price * rng.uniform(0.8, 1.2)
            mean_q = math.exp(a) * (price ** true_b)
            qty = max(0.1, rng.lognormal(mean=np.log(max(mean_q, 0.1)), sigma=0.25))
            unit_cost = price * rng.uniform(0.55, 0.75)
            rows.append(dict(product_id=sku, date=dt, price=round(price,2), quantity=round(qty,3), unit_cost=round(unit_cost,2)))
    return pd.DataFrame(rows)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### Data")
    mode = st.radio("Choose data:", ["Built-in demo", "Upload CSV"], index=0)
    currency = st.selectbox("Currency", ["R", "$", "€", "£"], index=0)
    st.markdown("---")
    st.markdown("### Aggregation")
    agg = st.selectbox("Aggregate to:", ["Daily","Weekly","Monthly"], index=1)
    st.caption("Prices averaged and quantities summed per period.")
    st.markdown("---")
    st.markdown("### Chart scaling")
    qty_norm = st.checkbox("Normalize Quantity to 0–1", value=False, help="Display only; model uses raw values.")
    money_scale = st.selectbox("Revenue/GP axis scaling", ["Raw", "Thousands (×1e3)", "Millions (×1e6)"], index=0)
    st.markdown("---")
    pct = st.slider("Price grid (±%)", 10, 80, 40, step=5)
    st.markdown("---")
    if st.button("Reset All"):
        st.experimental_rerun()

# ---------------- LOAD DATA ----------------
if mode == "Built-in demo":
    df = demo()
else:
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.info("Upload a CSV to proceed, or switch to the built-in demo in the sidebar.")
        st.stop()
    df = pd.read_csv(up)
df = normalize_columns(df)

# Aggregation
def to_period(d: pd.Series, level: str) -> pd.Series:
    if level == "Daily": return d.dt.to_period("D").dt.to_timestamp()
    if level == "Weekly": return d.dt.to_period("W-MON").dt.to_timestamp()
    if level == "Monthly": return d.dt.to_period("M").dt.to_timestamp()
    return d
df["date"] = to_period(df["date"], agg)

agg_map = {"price":"mean","quantity":"sum"}
if "unit_cost" in df.columns: agg_map["unit_cost"] = "mean"
work = df.groupby(["product_id","date"], as_index=False).agg(agg_map).sort_values(["product_id","date"])

# ---------------- SKU & FIT ----------------
st.markdown('<div class="section">Choose product</div>', unsafe_allow_html=True)
skus = work["product_id"].dropna().sort_values().unique().tolist()
sku = st.selectbox("SKU", skus, index=0)

sub = work[work["product_id"] == sku].copy()
fit = fit_loglog_elasticity(sub)
if not np.isfinite(fit.b):
    st.warning("Not enough valid observations to estimate elasticity for this SKU.")
    st.stop()

# Current (latest) price and unit cost
sub_sorted = sub.sort_values("date")
current_price = float(sub_sorted.iloc[-1]["price"])
uc = unit_cost_guess(sub["price"], sub["unit_cost"] if "unit_cost" in sub.columns else None)

# KPIs
c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f'<div class="kpi" style="border-left:6px solid #888;"><h3>Elasticity (b)</h3><p>{fit.b:.3f}</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi" style="border-left:6px solid #888;"><h3>Fit (R²)</h3><p>{fit.r2:.3f}</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi" style="border-left:6px solid #888;"><h3>Samples (n)</h3><p>{fit.n}</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="kpi" style="border-left:6px solid #888;"><h3>Class</h3><p>{classify(fit.b)}</p></div>', unsafe_allow_html=True)
st.markdown(f"**Current price** (latest): {money(current_price, currency)}  |  **Assumed unit cost**: {money(uc, currency)}")

# ---------------- PRICE SLIDER (LIVE) ----------------
median_price = float(sub["price"].median())
lo = max(0.01, median_price * (1 - pct/100))
hi = max(lo + 0.02, median_price * (1 + pct/100))

col_sl1, col_sl2 = st.columns([3,1])
with col_sl1:
    user_price = st.slider(
        "Choose a price to simulate",
        min_value=float(lo),
        max_value=float(hi),
        value=float(current_price),
        step=float((hi - lo)/100 or 0.10),
        help="Slide to adjust price; chart & KPIs update instantly."
    )
with col_sl2:
    # Live margin % at the chosen price
    margin_user_pct = (user_price - uc) / user_price * 100.0 if user_price > 0 else float('nan')
    st.metric("Margin @ price", f"{margin_user_pct:.1f}%")

# ---------------- MODEL & GRID ----------------
grid = np.linspace(lo, hi, 220)
q = demand(fit.a, fit.b, grid)
rev = grid * q
gp = (grid - uc) * q

# Scaling
scale_factor = 1.0
money_label_suffix = ""
if money_scale.startswith("Thousands"):
    scale_factor = 1e-3
    money_label_suffix = " (thousands)"
elif money_scale.startswith("Millions"):
    scale_factor = 1e-6
    money_label_suffix = " (millions)"
rev_s, gp_s = rev * scale_factor, gp * scale_factor

if qty_norm:
    q_min, q_max = float(np.min(q)), float(np.max(q))
    q_s = (q - q_min) / (q_max - q_min) if q_max > q_min else np.zeros_like(q)
    q_ylabel = "Quantity (normalized 0–1)"
else:
    q_s = q
    q_ylabel = "Quantity (Q)"

def pred_at(p: float) -> Tuple[float,float,float]:
    q_ = float(demand(fit.a, fit.b, np.array([p]))[0])
    r_ = p * q_
    g_ = (p - uc) * q_
    return q_, r_, g_

q_user, r_user, g_user = pred_at(user_price)
q_curr, r_curr, g_curr = pred_at(current_price)

# Optimal prices
opts = opt_prices_for_range(fit.a, fit.b, uc, lo, hi)
p_gp_interior = opts["p_gp_interior"]
p_gp_display = opts["p_gp_display"]
p_rev_boundary = opts["p_rev_boundary"]

# ---------------- TARGET MARGIN % SOLVER ----------------
st.markdown('<div class="section">Target Margin % Solver</div>', unsafe_allow_html=True)
tm_col1, tm_col2 = st.columns([1,2])
with tm_col1:
    target_margin_pct = st.slider("Target Gross Margin (%)", min_value=1, max_value=95, value=35, step=1,
                                  help="Solve for price P such that margin = (P − cost) / P.")
target_m = float(target_margin_pct) / 100.0
if target_m >= 1.0:
    st.warning("Target margin must be < 100%.")
    p_target = None
else:
    p_target = uc / (1.0 - target_m)  # P = c / (1 - m)

p_target_display = None
if p_target and np.isfinite(p_target) and p_target > 0:
    p_target_display = min(max(p_target, lo), hi)

if p_target is not None and np.isfinite(p_target):
    q_t, r_t, g_t = pred_at(p_target)
    with tm_col2:
        st.success(f"Suggested price for {target_margin_pct}% margin: {money(p_target, currency)} "
                   + ("" if abs(p_target - (p_target_display or p_target)) < 1e-9 else f" → in-range display: {money(p_target_display, currency)}"))
        st.markdown(
            f"- Predicted Q: **{q_t:,.2f}** &nbsp;&nbsp; "
            f"- Revenue: **{money(r_t, currency)}** &nbsp;&nbsp; "
            f"- GP: **{money(g_t, currency)}**"
        )
else:
    with tm_col2:
        st.info("Provide a valid target margin to compute price.")

# ---------------- ALL-IN-ONE CHART ----------------
st.markdown('<div class="section">All-in-One Chart (Quantity, Revenue & GP)</div>', unsafe_allow_html=True)
fig, ax1 = plt.subplots(figsize=(9.2, 5.6))

# Left y-axis — Quantity (blue)
l1, = ax1.plot(grid, q_s, label="Quantity (Q)", color='blue')
ax1.set_xlabel("Price")
ax1.set_ylabel(q_ylabel)
ax1.grid(True, linestyle="--", alpha=0.35)

# Right y-axis — Revenue (green) & GP (gold)
ax2 = ax1.twinx()
l2, = ax2.plot(grid, rev_s, label="Revenue (P×Q)"+money_label_suffix, color='green')
l3, = ax2.plot(grid, gp_s, label="Gross Profit ((P−cost)×Q)"+money_label_suffix, color='gold')
ax2.set_ylabel("Revenue / GP" + money_label_suffix)

# Vertical markers with colors
line_curr = ax1.axvline(current_price, linestyle='--', alpha=0.8, color='brown', label='Current Price')
line_user = ax1.axvline(user_price, linestyle='-.', alpha=0.8, color='green', label='User Price')
line_gp = None
if p_gp_display is not None:
    line_gp = ax1.axvline(p_gp_display, linestyle=':', alpha=0.8, color='orange', label='Opt GP (in-range)')
line_tm = None
if p_target_display is not None:
    line_tm = ax1.axvline(p_target_display, linestyle='--', alpha=0.8, color='purple', label='Target Margin Price')

# Helper to locate y at x (nearest in grid)
def y_at(x_arr, y_arr, x):
    idx = int(np.clip(np.searchsorted(x_arr, x), 0, len(x_arr)-1))
    return float(y_arr[idx])

# Scatter markers matching colors
def scatter_all(x, c_q, c_rev, c_gp):
    ax1.scatter([x], [y_at(grid, q_s, x)], s=60, zorder=3, color=c_q, edgecolor='white', linewidth=0.5)
    ax2.scatter([x], [y_at(grid, rev_s, x)], s=60, zorder=3, color=c_rev, edgecolor='white', linewidth=0.5)
    ax2.scatter([x], [y_at(grid, gp_s, x)], s=60, zorder=3, color=c_gp, edgecolor='white', linewidth=0.5)

scatter_all(current_price, 'blue', 'green', 'gold')
scatter_all(user_price, 'blue', 'green', 'gold')
if p_gp_display is not None:
    scatter_all(p_gp_display, 'blue', 'green', 'gold')
if p_target_display is not None:
    scatter_all(p_target_display, 'blue', 'green', 'gold')

# Legend (include vertical lines)
lines = [l1, l2, l3]
# Add dummies to include colored vlines:
lines += [ax1.axvline(current_price, color='brown', linestyle='--', alpha=0)]
lines += [ax1.axvline(user_price, color='green', linestyle='-.', alpha=0)]
if line_gp is not None:
    lines += [ax1.axvline(p_gp_display, color='orange', linestyle=':', alpha=0)]
if line_tm is not None:
    lines += [ax1.axvline(p_target_display, color='purple', linestyle='--', alpha=0)]
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc="upper right")

st.pyplot(fig, use_container_width=True)

# ---------------- KPIs ----------------
child5, child6, child7, child8 = st.columns(4)

with child5:
    st.markdown(f'<div class="kpi" style="border-left:6px solid blue;"><h3>Quantity</h3><p>{q_user:,.2f}</p></div>', unsafe_allow_html=True)

with child6:
    st.markdown(f'<div class="kpi" style="border-left:6px solid green;"><h3>Revenue</h3><p>{money(r_user, currency)}</p></div>', unsafe_allow_html=True)

with child7:
    st.markdown(f'<div class="kpi" style="border-left:6px solid gold;"><h3>Gross Profit</h3><p>{money(g_user, currency)}</p></div>', unsafe_allow_html=True)

if p_gp_interior is not None:
    _, _, g_opt = pred_at(p_gp_interior)
    with child8:
        st.markdown(f'<div class="kpi" style="border-left:6px solid orange;"><h3>Optimal GP</h3><p>{money(g_opt, currency)}</p></div>', unsafe_allow_html=True)
else:
    with child8:
        st.markdown(f'<div class="kpi" style="border-left:6px solid orange;"><h3>Revenue Boundary</h3><p>{money(p_rev_boundary, currency) if p_rev_boundary else "—"}</p></div>', unsafe_allow_html=True)

# Target margin KPI row
if p_target is not None and np.isfinite(p_target):
    qt, rt, gt = pred_at(p_target)
    cm1, cm2, cm3 = st.columns(3)
    with cm1: st.markdown(f'<div class="kpi" style="border-left:6px solid purple;"><h3>Target Margin</h3><p>{target_margin_pct}%</p></div>', unsafe_allow_html=True)
    with cm2: st.markdown(f'<div class="kpi" style="border-left:6px solid purple;"><h3>Price @ Target</h3><p>{money(p_target, currency)}</p></div>', unsafe_allow_html=True)
    with cm3: st.markdown(f'<div class="kpi" style="border-left:6px solid purple;"><h3>GP @ Target</h3><p>{money(gt, currency)}</p></div>', unsafe_allow_html=True)

st.caption("Notes: Interior GP-optimal price exists when elasticity b < −1. Otherwise GP/Revenue are boundary-driven in this simple model.")
st.caption("© 2025 Real Analytics 101 — PED All-in-One (Clean)")

if __name__ == "__main__":
    print("This is a Streamlit app. Run with:")
    print("  streamlit run 04_price_elasticity_ped.py")
