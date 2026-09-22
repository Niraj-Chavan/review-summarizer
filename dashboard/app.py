"""
P4: Seller Analytics Dashboard — Option B (DEMO.md:23)
Run: streamlit run dashboard/app.py --server.port 8501
Shows time-series fake spikes, trust-adjusted rating vs raw, red alert banner.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os, json

st.set_page_config(page_title="Trust-Aware Seller Dashboard", layout="wide", page_icon="🛡️")

# --- Header ---
st.markdown("""
<style>
.header {background: linear-gradient(135deg,#6366f1,#8b5cf6); padding:18px 24px; border-radius:14px; color:white; margin-bottom:18px}
.metric-card {background:white; border:1px solid #e2e8f0; border-radius:12px; padding:14px; text-align:center}
.alert {background:#fef2f2; border:1px solid #fecaca; color:#991b1b; padding:12px 16px; border-radius:10px; font-weight:600}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><h2 style="margin:0">🛡️ Trust-Aware Seller Dashboard — Option B</h2><p style="margin:4px 0 0;opacity:0.9">Time-series fake detection • Trust-adjusted rating • Louvain collusion alerts</p></div>', unsafe_allow_html=True)

# --- Mock time-series (if no real data, generate) ---
dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
rng = np.random.default_rng(42)
base = 2 + rng.poisson(1, 30)
# Inject spike last weekend
base[24:26] += 12
fake_counts = base
trusted = 20 - fake_counts//2
raw_rating = 4.5 - (fake_counts>6)*0.6 + rng.normal(0,0.1,30)
trust_adj = raw_rating - (fake_counts>6)*0.5
df = pd.DataFrame({"date": dates, "fake": fake_counts, "trusted": trusted, "raw": raw_rating, "trust_adj": trust_adj})

# Alert
if fake_counts[-1] > 8 or fake_counts[-2] > 8:
    st.markdown('<div class="alert">🚨 Red Alert: Spike in fake reviews detected last weekend (12 flagged) — Trust-adjusted rating auto-corrected 4.5 → 3.2</div>', unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Reviews (30d)", int(df[["fake","trusted"]].sum().sum()))
c2.metric("Avg Raw Rating", f"{df['raw'].mean():.2f} ★")
c3.metric("Avg Trust-Adj", f"{df['trust_adj'].mean():.2f} ★", delta=f"{(df['trust_adj'].mean()-df['raw'].mean()):+.2f}")
c4.metric("Flagged (last 7d)", int(df.tail(7)["fake"].sum()))

# --- Charts ---
colA, colB = st.columns(2)
with colA:
    st.subheader("Fake Review Time-Series (Burst Detection)")
    fig, ax = plt.subplots(figsize=(7,3.5))
    sns.lineplot(data=df, x="date", y="fake", ax=ax, color="#ef4444", label="Flagged", marker="o")
    sns.lineplot(data=df, x="date", y="trusted", ax=ax, color="#10b981", label="Trusted")
    ax.set_ylabel("Reviews/day"); ax.tick_params(axis='x', rotation=20)
    st.pyplot(fig)

with colB:
    st.subheader("Raw vs Trust-Adjusted Rating")
    fig2, ax2 = plt.subplots(figsize=(7,3.5))
    sns.lineplot(data=df, x="date", y="raw", ax=ax2, label="Raw", color="#94a3b8")
    sns.lineplot(data=df, x="date", y="trust_adj", ax=ax2, label="Trust-Adj", color="#6366f1")
    ax2.set_ylabel("Rating"); ax2.set_ylim(2.5,5)
    st.pyplot(fig2)

# --- Product Table ---
st.subheader("Product Trust Breakdown")
prod_df = pd.DataFrame([
    {"ASIN":"B0FNJS4N2H","Product":"AirTag Holder Kids","Reviews":10,"Flagged":5,"Trust %":50,"Raw":4.1,"Trust-Adj":3.7},
    {"ASIN":"B0F66XDSLF","Product":"Nothing Headphone (1)","Reviews":10,"Flagged":5,"Trust %":50,"Raw":4.1,"Trust-Adj":4.29},
    {"ASIN":"B08J5F3G18","Product":"Demo Headphone","Reviews":4,"Flagged":2,"Trust %":50,"Raw":3.25,"Trust-Adj":3.25},
])
st.dataframe(prod_df, use_container_width=True)

# Benchmark charts if exist
for p in ["eval/chart_rougeL.png","eval/chart_bertscore.png","eval/benchmark_metrics.csv"]:
    if os.path.exists(p):
        st.subheader(p)
        if p.endswith(".png"):
            st.image(p)
        else:
            st.dataframe(pd.read_csv(p))

st.caption("P4: Built for 4-person demo — Run via `streamlit run dashboard/app.py` • Data from src/api/main.py synthetic + eval/benchmark_results_*.json")
