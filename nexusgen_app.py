# =============================================================================
# NexusGen v2.0 – Streamlit Chatbot + Dashboard
# Gen-AI Supply Chain Oracle (Astra UD Trucks)
#
# Jalankan: streamlit run nexusgen_app.py
# Install : pip install streamlit scikit-learn scipy pandas numpy matplotlib
#           pip install prophet   ← untuk forecast lebih akurat (opsional)
# =============================================================================

import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusGen v2.0 – Supply Chain Oracle",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #080c12; color: #E8EAF6; }
[data-testid="stSidebar"]          { background: #0d1420; }
.stTabs [data-baseweb="tab-list"]  { background: #0d1420; }
.stTabs [data-baseweb="tab"]       { color: #8899AA; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #69F0AE; border-bottom: 2px solid #69F0AE; }
div[data-testid="metric-container"] { background: #0d1420; border: 1px solid #1e2a3a; border-radius: 8px; padding: 10px; }
.stButton>button { background: #1e2a3a; color: #69F0AE; border: 1px solid #69F0AE; border-radius: 6px; }
.stButton>button:hover { background: #69F0AE; color: #080c12; }
h1,h2,h3 { color: #69F0AE; }

/* Chat bubbles */
.chat-user {
    background: #1e2a3a; border-radius: 12px 12px 0 12px;
    padding: 10px 14px; margin: 6px 0; max-width: 75%;
    float: right; clear: both; color: #E8EAF6;
}
.chat-assistant {
    background: #0d1e0d; border: 1px solid #1a3a1a; border-radius: 12px 12px 12px 0;
    padding: 10px 14px; margin: 6px 0; max-width: 85%;
    float: left; clear: both; color: #E8EAF6;
}
.chat-system {
    background: #1a1a0d; border: 1px solid #3a3a1e;
    border-radius: 8px; padding: 8px 12px;
    margin: 4px 0; clear: both; color: #FFF176; font-size: 0.85rem;
}
.clearfix { clear: both; }
.api-active { color: #69F0AE; font-weight: bold; }
.api-inactive { color: #FF5252; font-weight: bold; }
.solution-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.8rem; font-weight: bold; margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# KONSTANTA & KATALOG
# ─────────────────────────────────────────────────────────────────
SPARE_PARTS_CATALOG: dict[str, dict] = {
    "Filter Oli HD":          {"base_prob": 0.45, "base_qty": 6,  "unit_cost": 85_000,    "criticality": "HIGH"},
    "Kampas Rem Besar":       {"base_prob": 0.30, "base_qty": 4,  "unit_cost": 320_000,   "criticality": "HIGH"},
    "Pompa Injeksi":          {"base_prob": 0.18, "base_qty": 2,  "unit_cost": 1_200_000, "criticality": "CRITICAL"},
    "Gasket Kepala Silinder": {"base_prob": 0.22, "base_qty": 5,  "unit_cost": 450_000,   "criticality": "MEDIUM"},
    "Bearing Differential":   {"base_prob": 0.12, "base_qty": 2,  "unit_cost": 780_000,   "criticality": "HIGH"},
    "V-Belt Kompressor":      {"base_prob": 0.35, "base_qty": 8,  "unit_cost": 95_000,    "criticality": "MEDIUM"},
    "Seal Kit Hidrolik":      {"base_prob": 0.20, "base_qty": 3,  "unit_cost": 230_000,   "criticality": "MEDIUM"},
    "Relay Control Unit":     {"base_prob": 0.08, "base_qty": 1,  "unit_cost": 2_100_000, "criticality": "CRITICAL"},
}

ORDER_COST_IDR    = 750_000
HOLDING_COST_PCT  = 0.22
CLAUDE_MODEL      = "claude-sonnet-4-20250514"


# ─────────────────────────────────────────────────────────────────
# DATA & ANALISIS (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_demand_data(n_months: int = 48, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    date_range = pd.date_range(start="2021-01-01", periods=n_months, freq="MS")
    records = []
    for part_name, params in SPARE_PARTS_CATALOG.items():
        for i, date in enumerate(date_range):
            seasonal = (1.9 if date.month in [11,12,1] else
                        1.4 if date.month in [6,7,8]   else 1.0)
            trend = 1 + 0.35 / (1 + np.exp(-0.15*(i - n_months/2)))
            prob  = min(params["base_prob"] * seasonal * trend, 0.97)
            occurs = np.random.binomial(1, prob)
            if occurs:
                r = max(1, int(params["base_qty"] * seasonal))
                qty = int(np.random.negative_binomial(r, r/(r + params["base_qty"]*seasonal + 1e-9))) + 1
            else:
                qty = 0
            records.append({
                "date": date, "part_name": part_name, "demand": qty,
                "month": date.month, "year": date.year, "quarter": date.quarter,
                "unit_cost": params["unit_cost"], "criticality": params["criticality"],
                "seasonal_factor": round(seasonal, 2),
            })
    return pd.DataFrame(records)


def forecast_demand(demand_series: np.ndarray, dates: pd.DatetimeIndex,
                    horizon: int = 6) -> dict:
    series = np.asarray(demand_series, dtype=float)

    # Prophet (jika tersedia)
    prophet_fc = None
    try:
        from prophet import Prophet
        df_p = pd.DataFrame({"ds": dates, "y": series})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="multiplicative",
                    changepoint_prior_scale=0.05, interval_width=0.80)
        m.fit(df_p, iter=200)
        future = m.make_future_dataframe(periods=horizon, freq="MS")
        fc = m.predict(future)
        prophet_fc = {
            "mean":  np.maximum(0, fc["yhat"].values[-horizon:]),
            "lower": np.maximum(0, fc["yhat_lower"].values[-horizon:]),
            "upper": np.maximum(0, fc["yhat_upper"].values[-horizon:]),
        }
    except Exception:
        pass

    # HGB dengan lag features
    n_lag = min(12, len(series)//2)
    if len(series) >= n_lag + 2:
        fd = pd.DataFrame({"demand": series})
        for lag in range(1, n_lag+1): fd[f"lag_{lag}"] = fd["demand"].shift(lag)
        fd["trend"]     = np.arange(len(fd), dtype=float)
        fd["month"]     = (np.arange(len(fd)) % 12) + 1
        fd["sin_month"] = np.sin(2*np.pi*fd["month"]/12)
        fd["cos_month"] = np.cos(2*np.pi*fd["month"]/12)
        fd["roll3"]     = fd["demand"].shift(1).rolling(3).mean()
        fd["roll6"]     = fd["demand"].shift(1).rolling(6).mean()
        fd = fd.dropna().reset_index(drop=True)
        fc_cols = [c for c in fd.columns if c != "demand"]
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=24, random_state=42)
        mdl.fit(fd[fc_cols].values, fd["demand"].values)
        last_w = list(series[-n_lag:])
        hgb_preds = []
        for step in range(horizon):
            t = float(len(series)+step); mv = float((len(series)+step)%12+1)
            r3 = float(np.mean(last_w[-3:])) if len(last_w)>=3 else float(np.mean(last_w))
            r6 = float(np.mean(last_w[-6:])) if len(last_w)>=6 else r3
            feats = last_w[-n_lag:] + [t, mv, np.sin(2*np.pi*mv/12), np.cos(2*np.pi*mv/12), r3, r6]
            pred = float(max(0, mdl.predict(np.array(feats).reshape(1,-1))[0]))
            hgb_preds.append(pred); last_w.append(pred)
        hgb_fc = np.array(hgb_preds)
    else:
        hgb_fc = np.full(horizon, float(series.mean()))

    # Croston SBA
    d = series.copy(); nz = np.where(d > 0)[0]
    if len(nz) > 0:
        at = d[nz].mean(); pt = float(np.diff(np.concatenate(([-1], nz))).mean())
        for i, v in enumerate(d):
            if v > 0:
                at = 0.1*v + 0.9*at
                pt = 0.1*max(1,i) + 0.9*pt
        croston_next = float((1 - 0.05)*at / max(pt, 1e-6))
    else:
        croston_next = 0.0

    if prophet_fc is not None:
        ensemble = 0.6*prophet_fc["mean"] + 0.4*hgb_fc
        method   = "Prophet(0.6) + HGB(0.4)"
        lower, upper = prophet_fc["lower"], prophet_fc["upper"]
    else:
        ensemble = hgb_fc
        method   = "HGB Boosting (pip install prophet untuk lebih akurat)"
        std_fc   = hgb_fc.std()*0.5
        lower    = np.maximum(0, hgb_fc - std_fc)
        upper    = hgb_fc + std_fc

    return {
        "ensemble_forecast": np.maximum(0, ensemble),
        "hgb_forecast":      hgb_fc,
        "prophet_forecast":  prophet_fc,
        "croston_next":      croston_next,
        "lower_bound":       lower,
        "upper_bound":       upper,
        "method":            method,
        "horizon":           horizon,
    }


def calc_inventory(demand_series, part_name="", unit_cost=0,
                   criticality="MEDIUM", lead_time=2, service_level=0.97) -> dict:
    demand = np.asarray([d for d in demand_series if d >= 0], dtype=float)
    if len(demand) == 0: demand = np.array([1.0])
    sl_map = {"CRITICAL": 0.99, "HIGH": 0.97, "MEDIUM": 0.95}
    adj_sl = sl_map.get(criticality, service_level)
    Z      = float(stats.norm.ppf(adj_sl))
    mean_d = float(demand.mean()); std_d = float(demand.std()) if len(demand)>1 else 0.5
    cv     = std_d / (mean_d + 1e-9)
    xyz    = "X" if cv < 0.5 else "Y" if cv < 1.0 else "Z"
    annual_val = mean_d * 12 * unit_cost
    abc    = "A" if annual_val > 5_000_000 else "B" if annual_val > 1_000_000 else "C"
    std_lt = lead_time * 0.2
    ss     = int(np.ceil(np.sqrt((Z*std_d*np.sqrt(lead_time))**2 + (Z*mean_d*std_lt)**2)))
    ss     = max(1, ss)
    rop    = int(np.ceil(mean_d*lead_time + ss))
    h      = (unit_cost * HOLDING_COST_PCT) if unit_cost > 0 else 50_000
    eoq    = int(np.ceil(np.sqrt(2*mean_d*12*ORDER_COST_IDR/(h+1e-9)))); eoq = max(1,eoq)
    return {
        "part_name": part_name, "criticality": criticality,
        "abc_class": abc, "xyz_class": xyz,
        "service_level_pct": f"{adj_sl*100:.0f}%",
        "Z": round(Z,4), "mean_demand": round(mean_d,2),
        "std_demand": round(std_d,3), "cv": round(cv,3),
        "lead_time": lead_time, "safety_stock": ss, "reorder_point": rop, "eoq": eoq,
        "annual_holding_cost_idr": int((ss+eoq/2)*h),
        "inventory_value_idr": int(rop*unit_cost),
        "stockout_risk_pct": f"{(1-adj_sl)*100:.1f}%",
        "unit_cost": int(unit_cost),
    }


def calc_risk(current_stock, inv_plan, forecast_result) -> dict:
    ss   = inv_plan["safety_stock"]; rop = inv_plan["reorder_point"]
    crit = inv_plan["criticality"]
    cw   = {"CRITICAL":1.5,"HIGH":1.2,"MEDIUM":1.0}.get(crit,1.0)
    stock_ratio = current_stock / (rop+1e-9)
    inv_risk    = int(np.clip(max(0,(1-stock_ratio)*60)*cw + (15 if current_stock<=ss else 0), 0, 100))
    fc = forecast_result["ensemble_forecast"]
    cv_fc = float(np.std(fc)/(np.mean(fc)+1e-9))
    demand_risk = int(np.clip(cv_fc*80, 0, 100))
    avg_fc = float(np.mean(fc))
    months_cover = current_stock/(avg_fc+1e-9)
    cover_risk  = int(np.clip(max(0,(3-months_cover)/3*60)*cw, 0, 100))
    composite   = int(0.40*inv_risk + 0.30*demand_risk + 0.30*cover_risk)
    level = ("🔴 KRITIS" if composite>=70 else "🟠 TINGGI" if composite>=45
             else "🟡 SEDANG" if composite>=25 else "🟢 RENDAH")
    return {"composite_score":composite,"level":level,"inventory_risk":inv_risk,
            "demand_risk":demand_risk,"coverage_risk":cover_risk,
            "months_cover":round(months_cover,1)}


# ─────────────────────────────────────────────────────────────────
# MULTI-SOLUTION GENERATOR
# ─────────────────────────────────────────────────────────────────

def generate_multi_solutions(demand_series, unit_cost, criticality,
                              current_stock, forecast_result) -> list[dict]:
    """
    Hasilkan 5 solusi optimal dengan prioritas berbeda-beda.
    Setiap solusi punya trade-off yang berbeda antara:
    service level, safety stock, biaya, dan risiko stockout.
    """
    base_fc = float(np.mean(forecast_result["ensemble_forecast"]))
    lead_time_options = [1, 2, 3]
    sl_options = {
        "CRITICAL": [0.97, 0.99, 0.999],
        "HIGH":     [0.95, 0.97, 0.99],
        "MEDIUM":   [0.90, 0.95, 0.97],
    }
    sls = sl_options.get(criticality, [0.90, 0.95, 0.97])

    profiles = [
        {
            "name":   "💰 Hemat Biaya Maksimal",
            "desc":   "Minimasi biaya holding & order. Cocok untuk part non-kritis dengan demand stabil.",
            "color":  "#FFB74D",
            "sl":     sls[0],
            "lt":     lead_time_options[2],  # lead time panjang → order lebih jarang
            "icon":   "💰",
        },
        {
            "name":   "⚖️ Seimbang (Recommended)",
            "desc":   "Keseimbangan optimal antara biaya, service level, dan risiko. Pilihan default.",
            "color":  "#69F0AE",
            "sl":     sls[1],
            "lt":     lead_time_options[1],
            "icon":   "⚖️",
        },
        {
            "name":   "🛡️ Zero-Stockout",
            "desc":   "Prioritas absolut pada ketersediaan stok. Biaya lebih tinggi tapi risiko downtime minimal.",
            "color":  "#4FC3F7",
            "sl":     sls[2],
            "lt":     lead_time_options[0],  # lead time pendek → respons cepat
            "icon":   "🛡️",
        },
        {
            "name":   "🚀 Respons Cepat",
            "desc":   "Lead time sangat pendek (supplier lokal/express). Cocok untuk situasi darurat produksi.",
            "color":  "#CE93D8",
            "sl":     sls[1],
            "lt":     1,
            "icon":   "🚀",
        },
        {
            "name":   "📦 Bulk Economy",
            "desc":   "EOQ besar, order jarang. Cocok untuk part dengan demand stabil dan storage murah.",
            "color":  "#FFF176",
            "sl":     sls[0],
            "lt":     lead_time_options[1],
            "icon":   "📦",
            "eoq_multiplier": 1.5,
        },
    ]

    solutions = []
    for p in profiles:
        inv = calc_inventory(demand_series, unit_cost=unit_cost,
                             criticality=criticality,
                             lead_time=p["lt"], service_level=p["sl"])
        eoq_final = int(inv["eoq"] * p.get("eoq_multiplier", 1.0))
        h = unit_cost * HOLDING_COST_PCT if unit_cost > 0 else 50_000
        annual_order_cost   = (base_fc*12 / max(eoq_final,1)) * ORDER_COST_IDR
        annual_holding_cost = (inv["safety_stock"] + eoq_final/2) * h
        total_annual_cost   = annual_order_cost + annual_holding_cost
        months_cover        = current_stock / (base_fc + 1e-9)
        decision = (
            "🔴 PESAN DARURAT" if current_stock <= 0 else
            "🔴 BELI SEKARANG" if current_stock <= inv["safety_stock"] else
            "🟠 PESAN SEGERA"  if current_stock <= inv["reorder_point"] else
            "🟡 RENCANAKAN PO" if months_cover < 3 else
            "🟢 TAHAN STOK"
        )
        solutions.append({
            **p,
            "inv":               inv,
            "eoq_final":         eoq_final,
            "annual_order_cost": int(annual_order_cost),
            "annual_holding":    int(annual_holding_cost),
            "total_annual_cost": int(total_annual_cost),
            "months_cover":      round(months_cover, 1),
            "decision":          decision,
            "service_level_pct": f"{p['sl']*100:.0f}%",
        })

    return solutions


# ─────────────────────────────────────────────────────────────────
# CLAUDE AI CHATBOT
# ─────────────────────────────────────────────────────────────────

def call_claude(messages_history: list[dict], api_key: str,
                system_prompt: str = "") -> str:
    """
    Panggil Claude API dengan riwayat percakapan penuh (multi-turn).
    messages_history: list of {"role": "user"/"assistant", "content": "..."}
    """
    try:
        import urllib.request

        payload = {
            "model":      CLAUDE_MODEL,
            "max_tokens": 1200,
            "messages":   messages_history,
        }
        if system_prompt:
            payload["system"] = system_prompt

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"]
    except Exception as e:
        return f"❌ Error memanggil Claude API: {str(e)}"


def build_system_prompt(all_results: dict, current_stock: int,
                         selected_part: str) -> str:
    """
    System prompt lengkap berisi seluruh konteks analisis supply chain.
    Claude akan memiliki akses ke semua data saat menjawab pertanyaan.
    """
    ctx_parts = []
    for pname, res in all_results.items():
        inv = res["inv"]; fc = res["fc"]; risk = res["risk"]
        ctx_parts.append(
            f"- {pname}: ABC={inv['abc_class']} XYZ={inv['xyz_class']} "
            f"crit={inv['criticality']} SS={inv['safety_stock']} "
            f"ROP={inv['reorder_point']} EOQ={inv['eoq']} "
            f"forecast_avg={round(float(np.mean(fc['ensemble_forecast'])),1)}/bln "
            f"risk={risk['composite_score']}/100 ({risk['level']})"
        )

    sel_res = all_results[selected_part]
    sel_inv = sel_res["inv"]; sel_fc = sel_res["fc"]; sel_risk = sel_res["risk"]
    fc_vals = [round(v,1) for v in sel_fc["ensemble_forecast"].tolist()]

    return f"""Kamu adalah NexusGen Supply Chain Oracle v2, AI advisor ahli manajemen suku cadang truk berat untuk Astra UD Trucks.
Kamu memiliki akses ke seluruh data analisis supply chain berikut dan harus menjawab dalam Bahasa Indonesia.

KONTEKS SISTEM:
- Platform: NexusGen v2.0 dengan Ensemble Forecasting (Prophet + HGB)
- Periode data: 48 bulan historis, 8 jenis suku cadang
- Mesin forecast: {sel_fc['method']}

DATA SEMUA SUKU CADANG:
{chr(10).join(ctx_parts)}

FOKUS ANALISIS – {selected_part}:
- Stok saat ini  : {current_stock} unit
- Criticality    : {sel_inv['criticality']} | ABC: {sel_inv['abc_class']} | XYZ: {sel_inv['xyz_class']}
- CV Demand      : {sel_inv['cv']} | Mean: {sel_inv['mean_demand']}/bln
- Safety Stock   : {sel_inv['safety_stock']} unit (dynamic SS dengan variance LT)
- Reorder Point  : {sel_inv['reorder_point']} unit
- EOQ            : {sel_inv['eoq']} unit
- Forecast 6 bln : {fc_vals}
- Risk Score     : {sel_risk['composite_score']}/100 ({sel_risk['level']})
  ├─ Inventory   : {sel_risk['inventory_risk']}/100
  ├─ Demand      : {sel_risk['demand_risk']}/100
  └─ Coverage    : {sel_risk['coverage_risk']}/100
- Sisa cover     : {sel_risk['months_cover']} bulan

PANDUAN MENJAWAB:
- Jawab singkat, padat, actionable
- Selalu berikan angka konkret (unit, Rp, bulan)
- Jika ditanya suku cadang lain, gunakan data di atas
- Jika ditanya strategi umum, berikan framework yang relevan
- Gunakan emoji secara bijak untuk keterbacaan
- Akhiri jawaban dengan 1 pertanyaan konfirmasi jika konteks tidak cukup"""


# ─────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────

for key, default in {
    "initialized":    False,
    "df":             None,
    "all_results":    None,
    "chat_history":   [],       # {"role": ..., "content": ...}
    "api_key":        "",
    "api_verified":   False,
    "selected_part":  list(SPARE_PARTS_CATALOG.keys())[0],
    "current_stock":  4,
    "n_months":       48,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/truck.png", width=60)
    st.markdown("## 📦 NexusGen v2.0")
    st.markdown("*Gen-AI Supply Chain Oracle*")
    st.divider()

    # ── API KEY ──
    st.markdown("### 🤖 Claude AI Configuration")
    api_input = st.text_input(
        "Masukkan Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        value=st.session_state.api_key,
        help="Dapatkan API key di console.anthropic.com",
    )

    verify_btn = st.button("✅ Verifikasi & Aktifkan Chatbot", use_container_width=True)

    if verify_btn and api_input.strip():
        with st.spinner("Memverifikasi API key..."):
            test_msg = call_claude(
                [{"role":"user","content":"Balas hanya dengan kata: AKTIF"}],
                api_key=api_input.strip(),
            )
        if "AKTIF" in test_msg or "aktif" in test_msg.lower():
            st.session_state.api_key    = api_input.strip()
            st.session_state.api_verified = True
            st.success("✅ API Key valid! Chatbot aktif.")
        elif "❌" in test_msg:
            st.error(f"API Key tidak valid: {test_msg}")
            st.session_state.api_verified = False
        else:
            # Anggap berhasil jika ada respons (bukan error)
            st.session_state.api_key    = api_input.strip()
            st.session_state.api_verified = True
            st.success("✅ Chatbot aktif!")

    if st.session_state.api_verified:
        st.markdown('<span class="api-active">🟢 Chatbot: AKTIF</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="api-inactive">🔴 Chatbot: NONAKTIF (masukkan API key)</span>',
                    unsafe_allow_html=True)

    st.divider()

    # ── PARAMETER ──
    st.markdown("### ⚙️ Parameter Analisis")
    st.session_state.n_months = st.selectbox("Periode Data", [24, 36, 48], index=2,
                                              format_func=lambda x: f"{x} bulan")
    st.session_state.selected_part = st.selectbox(
        "Suku Cadang Fokus", list(SPARE_PARTS_CATALOG.keys()), index=1
    )
    st.session_state.current_stock = st.number_input(
        "Stok Saat Ini (unit)", min_value=0, max_value=200,
        value=st.session_state.current_stock, step=1
    )
    forecast_horizon = st.slider("Horizon Forecast (bulan)", 3, 12, 6)

    st.divider()
    run_btn = st.button("🚀 Jalankan Analisis", use_container_width=True)

    if st.session_state.initialized and st.session_state.api_verified:
        st.divider()
        st.markdown("### 💬 Quick Questions")
        quick_qs = [
            "Apa keputusan pembelian terpenting sekarang?",
            "Part mana yang paling berisiko?",
            "Berikan strategi penghematan biaya inventory",
            "Jelaskan ABC-XYZ untuk semua part",
            "Kapan harus order dan berapa kuantitasnya?",
        ]
        for q in quick_qs:
            if st.button(q[:38]+"…" if len(q)>38 else q, key=f"quick_{q[:10]}",
                         use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})
                st.session_state._pending_quick = True
                st.rerun()


# ─────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#69F0AE; font-size:2.2rem; margin-bottom:0'>
📦 NexusGen v2.0 — Supply Chain Oracle
</h1>
<p style='text-align:center; color:#8899AA; margin-top:4px'>
Astra UD Trucks | Prophet + HGB Ensemble · Dynamic SS (ABC-XYZ) · Multi-Solusi · Claude AI Chatbot
</p>
""", unsafe_allow_html=True)
st.divider()


# ─────────────────────────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────────────────────────
if run_btn or not st.session_state.initialized:
    with st.spinner("📦 Memproses data demand & membangun model forecast..."):
        df = generate_demand_data(n_months=st.session_state.n_months)
        all_results = {}
        for pname, pinfo in SPARE_PARTS_CATALOG.items():
            p_df      = df[df["part_name"]==pname].sort_values("date")
            d_series  = p_df["demand"].values
            d_dates   = pd.DatetimeIndex(p_df["date"].values)
            fc        = forecast_demand(d_series, d_dates, horizon=forecast_horizon)
            inv       = calc_inventory(d_series, part_name=pname,
                                       unit_cost=pinfo["unit_cost"],
                                       criticality=pinfo["criticality"])
            risk      = calc_risk(st.session_state.current_stock, inv, fc)
            all_results[pname] = {"fc":fc,"inv":inv,"risk":risk,"demand":d_series,"dates":d_dates}

        st.session_state.df          = df
        st.session_state.all_results = all_results
        st.session_state.initialized = True

        # Reset chat saat analisis ulang (dengan pesan sambutan)
        sel  = st.session_state.selected_part
        sres = all_results[sel]
        welcome = (f"Halo! Saya **NexusGen AI** siap membantu analisis supply chain Anda. 👋\n\n"
                   f"Analisis sudah dimuat untuk **{len(SPARE_PARTS_CATALOG)} suku cadang** "
                   f"dengan data **{st.session_state.n_months} bulan**.\n\n"
                   f"Fokus saat ini: **{sel}** | "
                   f"Risk: {sres['risk']['composite_score']}/100 ({sres['risk']['level']}) | "
                   f"Stok: {st.session_state.current_stock} unit → "
                   f"Cover: {sres['risk']['months_cover']} bulan\n\n"
                   f"{'🟢 Chatbot aktif dengan Claude AI.' if st.session_state.api_verified else '🔴 Masukkan API key di sidebar untuk mengaktifkan chatbot Claude AI.'}\n\n"
                   f"Apa yang ingin Anda tanyakan?")
        st.session_state.chat_history = [{"role":"assistant","content":welcome}]

    st.success(f"✅ Analisis selesai! {len(SPARE_PARTS_CATALOG)} parts dianalisis.")


if not st.session_state.initialized:
    st.info("👈 Klik **Jalankan Analisis** di sidebar untuk memulai.")
    st.stop()

df          = st.session_state.df
all_results = st.session_state.all_results
sel_part    = st.session_state.selected_part
cur_stock   = st.session_state.current_stock
sel_res     = all_results[sel_part]
sel_inv     = sel_res["inv"]
sel_fc      = sel_res["fc"]
sel_risk    = sel_res["risk"]


# ─────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("📦 Part Fokus",   sel_part[:16]+"…" if len(sel_part)>16 else sel_part)
k2.metric("🏷️ ABC-XYZ",     sel_inv["abc_class"]+sel_inv["xyz_class"])
k3.metric("⚠️ Risk Score",   f"{sel_risk['composite_score']}/100",  sel_risk["level"])
k4.metric("🛡️ Safety Stock", f"{sel_inv['safety_stock']} unit")
k5.metric("📍 ROP",          f"{sel_inv['reorder_point']} unit")
k6.metric("📅 Cover",        f"{sel_risk['months_cover']} bulan")


# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Chatbot AI",
    "🎯 Multi-Solusi Optimal",
    "🔮 Forecast & Demand",
    "📊 Dashboard Semua Part",
    "⚙️ Detail Inventori",
])


# ════════════════════════════════════════════════════════════════
# TAB 1: CHATBOT AI
# ════════════════════════════════════════════════════════════════
with tab1:
    if not st.session_state.api_verified:
        st.warning(
            "🔐 **Chatbot belum aktif.** Masukkan Anthropic API Key di sidebar lalu klik "
            "**Verifikasi & Aktifkan Chatbot**."
        )
        st.markdown("""
        **Cara mendapatkan API Key:**
        1. Buka [console.anthropic.com](https://console.anthropic.com)
        2. Masuk / daftar akun
        3. Klik **API Keys** → **Create Key**
        4. Salin key dan tempel di sidebar
        """)
        st.info("💡 Saat chatbot belum aktif, Anda masih bisa menggunakan tab lainnya "
                "untuk melihat dashboard, forecast, dan multi-solusi.")
    else:
        st.markdown(f"### 💬 Chat dengan NexusGen AI — Fokus: **{sel_part}**")
        st.markdown(
            "Tanyakan apa saja seputar manajemen suku cadang, strategi pembelian, "
            "analisis risiko, atau optimasi inventori."
        )

        # Tampilkan riwayat chat
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "assistant":
                    st.markdown(
                        f'<div class="chat-assistant">🤖 <b>NexusGen AI</b><br>{msg["content"]}</div>'
                        '<div class="clearfix"></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-user">👤 <b>Anda</b><br>{msg["content"]}</div>'
                        '<div class="clearfix"></div>',
                        unsafe_allow_html=True
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # Input chat
        col_inp, col_send, col_clear = st.columns([6, 1, 1])
        with col_inp:
            user_input = st.text_input(
                "Ketik pertanyaan Anda...",
                key="chat_input",
                label_visibility="collapsed",
                placeholder="Contoh: Kapan saya harus order Kampas Rem Besar?",
            )
        with col_send:
            send_btn = st.button("📤 Kirim", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Reset", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Proses pesan baru (dari input atau quick question)
        pending_input = None
        if send_btn and user_input.strip():
            pending_input = user_input.strip()
            st.session_state.chat_history.append({"role":"user","content":pending_input})

        if getattr(st.session_state, "_pending_quick", False):
            st.session_state._pending_quick = False
            pending_input = st.session_state.chat_history[-1]["content"]

        if pending_input:
            with st.spinner("🤖 NexusGen AI sedang menganalisis..."):
                sys_prompt = build_system_prompt(
                    all_results, cur_stock, sel_part
                )
                # Kirim riwayat percakapan penuh (multi-turn)
                response = call_claude(
                    messages_history=st.session_state.chat_history,
                    api_key=st.session_state.api_key,
                    system_prompt=sys_prompt,
                )
                st.session_state.chat_history.append({"role":"assistant","content":response})
            st.rerun()

        # Suggestions kontekstual
        st.markdown("---")
        st.markdown("**💡 Pertanyaan yang disarankan berdasarkan kondisi saat ini:**")
        risk_score = sel_risk["composite_score"]
        suggestions = []
        if cur_stock <= sel_inv["safety_stock"]:
            suggestions.append(f"Stok {sel_part} sudah di bawah safety stock, apa yang harus saya lakukan segera?")
        if risk_score >= 60:
            suggestions.append(f"Risk score {risk_score}/100 tergolong tinggi, apa langkah mitigasinya?")
        suggestions += [
            f"Bandingkan solusi 'Hemat Biaya' vs 'Zero-Stockout' untuk {sel_part}",
            "Part mana yang perlu saya prioritaskan untuk dipesan bulan ini?",
            "Bagaimana cara mengurangi total biaya inventori tanpa mengorbankan service level?",
        ]
        sc_cols = st.columns(min(len(suggestions), 3))
        for i, (col, sug) in enumerate(zip(sc_cols, suggestions[:3])):
            with col:
                if st.button(sug[:45]+"…" if len(sug)>45 else sug,
                             key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":sug})
                    st.session_state._pending_quick = True
                    st.rerun()


# ════════════════════════════════════════════════════════════════
# TAB 2: MULTI-SOLUSI OPTIMAL
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"## 🎯 5 Solusi Optimal — {sel_part}")
    st.markdown(
        "Setiap solusi dihasilkan dari kombinasi **service level**, **lead time**, dan **EOQ** "
        "yang berbeda. Pilih sesuai prioritas operasional Anda."
    )

    pinfo    = SPARE_PARTS_CATALOG[sel_part]
    d_series = df[df["part_name"]==sel_part].sort_values("date")["demand"].values
    multi_solutions = generate_multi_solutions(
        d_series, pinfo["unit_cost"], pinfo["criticality"],
        cur_stock, sel_fc
    )

    # Cards solusi
    for sol in multi_solutions:
        with st.container():
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns([2.5, 1, 1, 1, 1.5, 1.5])
            with sc1:
                st.markdown(
                    f'<div style="background:#0d1420;border:1px solid {sol["color"]};'
                    f'border-radius:10px;padding:10px;">'
                    f'<b style="color:{sol["color"]}">{sol["name"]}</b><br>'
                    f'<span style="color:#8899AA;font-size:0.82rem">{sol["desc"]}</span>'
                    f'</div>', unsafe_allow_html=True
                )
            sc2.metric("🛡️ SS",     f"{sol['inv']['safety_stock']} unit",
                       f"SL {sol['service_level_pct']}")
            sc3.metric("📍 ROP",    f"{sol['inv']['reorder_point']} unit",
                       f"LT {sol['lt']} bln")
            sc4.metric("📦 EOQ",    f"{sol['eoq_final']} unit")
            sc5.metric("💸 Biaya/Thn",
                       f"Rp {sol['total_annual_cost']/1e6:.1f}JT",
                       f"Hold: Rp {sol['annual_holding']/1e6:.1f}JT")
            sc6.metric("Keputusan", sol["decision"])
        st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    # Chart perbandingan
    st.markdown("### 📊 Perbandingan Biaya & Safety Stock")
    fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(13, 4), facecolor="#0d1420")
    for ax in axes_cmp: ax.set_facecolor("#080c12")

    sol_names_s  = [s["name"].split("(")[0].strip()[:20] for s in multi_solutions]
    total_costs  = [s["total_annual_cost"]/1e6 for s in multi_solutions]
    ss_vals_s    = [s["inv"]["safety_stock"] for s in multi_solutions]
    clrs_s       = [s["color"] for s in multi_solutions]

    b1s = axes_cmp[0].bar(sol_names_s, total_costs, color=clrs_s, alpha=0.85)
    for b, v in zip(b1s, total_costs):
        axes_cmp[0].text(b.get_x()+b.get_width()/2, v+0.05, f"Rp{v:.1f}JT",
                          ha="center", color="#E8EAF6", fontsize=8)
    axes_cmp[0].tick_params(colors="#E8EAF6", labelsize=7)
    axes_cmp[0].set_xticklabels(sol_names_s, rotation=20, ha="right")
    for sp in axes_cmp[0].spines.values(): sp.set_edgecolor("#1e2a3a")
    axes_cmp[0].set_ylabel("Juta Rupiah/Tahun", color="#E8EAF6")
    axes_cmp[0].set_title("Total Biaya Tahunan", color="#E8EAF6", fontsize=10)

    b2s = axes_cmp[1].bar(sol_names_s, ss_vals_s, color=clrs_s, alpha=0.85)
    for b, v in zip(b2s, ss_vals_s):
        axes_cmp[1].text(b.get_x()+b.get_width()/2, v+0.2, f"{v}u",
                          ha="center", color="#E8EAF6", fontsize=8)
    axes_cmp[1].tick_params(colors="#E8EAF6", labelsize=7)
    axes_cmp[1].set_xticklabels(sol_names_s, rotation=20, ha="right")
    for sp in axes_cmp[1].spines.values(): sp.set_edgecolor("#1e2a3a")
    axes_cmp[1].set_ylabel("Unit", color="#E8EAF6")
    axes_cmp[1].set_title("Safety Stock per Solusi", color="#E8EAF6", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close()

    st.markdown("### 📋 Tabel Ringkasan Semua Solusi")
    tbl_rows = [{
        "Solusi":          s["name"],
        "Service Level":   s["service_level_pct"],
        "Lead Time":       f"{s['lt']} bln",
        "Safety Stock":    f"{s['inv']['safety_stock']} unit",
        "ROP":             f"{s['inv']['reorder_point']} unit",
        "EOQ":             f"{s['eoq_final']} unit",
        "Biaya Hold/Thn":  f"Rp {s['annual_holding']/1e6:.1f}JT",
        "Biaya Order/Thn": f"Rp {s['annual_order_cost']/1e6:.1f}JT",
        "Total/Thn":       f"Rp {s['total_annual_cost']/1e6:.1f}JT",
        "Keputusan":       s["decision"],
    } for s in multi_solutions]
    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 3: FORECAST & DEMAND
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"## 🔮 Forecast & Analisis Demand — {sel_part}")
    st.markdown(f"**Metode:** `{sel_fc['method']}`")

    part_hist = df[df["part_name"]==sel_part].sort_values("date")
    hist_demand = part_hist["demand"].values
    hist_dates  = part_hist["date"].values
    fc_vals_arr = sel_fc["ensemble_forecast"]

    fig_fc, ax_fc = plt.subplots(figsize=(14, 4.5), facecolor="#0d1420")
    ax_fc.set_facecolor("#080c12")
    last_n = min(18, len(hist_demand))
    x_hist = np.arange(last_n)
    x_fc_p = np.arange(last_n, last_n + len(fc_vals_arr))
    ax_fc.bar(x_hist, hist_demand[-last_n:], color="#4FC3F7", alpha=0.7, width=0.7, label="Aktual")
    ax_fc.plot(x_hist, pd.Series(hist_demand[-last_n:]).rolling(3, min_periods=1).mean(),
               color="#FFF176", lw=1.5, label="Rolling 3-bln")
    ax_fc.bar(x_fc_p, fc_vals_arr, color="#69F0AE", alpha=0.8, width=0.7, label="Forecast")
    ax_fc.fill_between(x_fc_p, sel_fc["lower_bound"], sel_fc["upper_bound"],
                        alpha=0.2, color="#69F0AE", label="80% CI")
    ax_fc.axhline(sel_inv["safety_stock"],  color="#FF5252",  ls="--", lw=1.5,
                  label=f"SS={sel_inv['safety_stock']}")
    ax_fc.axhline(sel_inv["reorder_point"], color="#FFB74D",  ls="--", lw=1.5,
                  label=f"ROP={sel_inv['reorder_point']}")
    ax_fc.axvline(last_n - 0.5, color="#8899AA", ls=":", lw=1)
    ax_fc.text(last_n - 0.5 + 0.1, ax_fc.get_ylim()[1]*0.95, "▶ Forecast",
               color="#8899AA", fontsize=9)
    for sp in ax_fc.spines.values(): sp.set_edgecolor("#1e2a3a")
    ax_fc.tick_params(colors="#E8EAF6")
    ax_fc.set_ylabel("Qty", color="#E8EAF6")
    ax_fc.legend(facecolor="#0d1420", labelcolor="#E8EAF6", fontsize=8, ncol=3)
    ax_fc.set_title(f"Demand History + Forecast 6 Bulan — {sel_part}", color="#E8EAF6", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig_fc, use_container_width=True)
    plt.close()

    fc_col1, fc_col2, fc_col3 = st.columns(3)
    fc_col1.metric("Forecast Rata-rata", f"{float(np.mean(fc_vals_arr)):.1f} unit/bln")
    fc_col2.metric("Croston Baseline",   f"{sel_fc['croston_next']:.2f} unit/bln")
    fc_col3.metric("Puncak Forecast",    f"{float(fc_vals_arr.max()):.1f} unit")

    st.markdown("#### 📅 Detail Forecast per Bulan")
    fc_tbl = pd.DataFrame({
        "Bulan":        [f"M+{i+1}" for i in range(len(fc_vals_arr))],
        "Forecast":     [round(v,1) for v in fc_vals_arr],
        "Batas Bawah":  [round(v,1) for v in sel_fc["lower_bound"]],
        "Batas Atas":   [round(v,1) for v in sel_fc["upper_bound"]],
        "Status":       ["⚠️ Di bawah SS" if v < sel_inv["safety_stock"]
                         else "✅ Aman" for v in fc_vals_arr],
    })
    st.dataframe(fc_tbl, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 4: DASHBOARD SEMUA PART
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📊 Dashboard Semua Suku Cadang")

    # Summary table
    summary_rows = []
    for pname, res in all_results.items():
        inv  = res["inv"]; fc = res["fc"]; risk = res["risk"]
        summary_rows.append({
            "Suku Cadang":    pname,
            "Criticality":   inv["criticality"],
            "ABC-XYZ":       inv["abc_class"]+inv["xyz_class"],
            "Mean Demand":   f"{inv['mean_demand']:.1f}/bln",
            "CV":            inv["cv"],
            "Safety Stock":  inv["safety_stock"],
            "ROP":           inv["reorder_point"],
            "EOQ":           inv["eoq"],
            "Fc Avg/Bln":    round(float(np.mean(fc["ensemble_forecast"])),1),
            "Risk Score":    risk["composite_score"],
            "Risk Level":    risk["level"],
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df.sort_values("Risk Score", ascending=False),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # Visualisasi
    fig_all = plt.figure(figsize=(18, 10), facecolor="#0d1420")
    gs_all  = gridspec.GridSpec(2, 3, figure=fig_all, hspace=0.45, wspace=0.38)

    C = {"blue":"#4FC3F7","green":"#69F0AE","red":"#FF5252",
         "orange":"#FFB74D","purple":"#CE93D8","teal":"#4DB6AC",
         "yellow":"#FFF176","bg":"#0d1420","text":"#E8EAF6","border":"#1e2a3a"}

    def sax(ax, title=""):
        ax.set_facecolor(C["bg"]); ax.tick_params(colors=C["text"], labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(C["border"])
        if title: ax.set_title(title, color=C["text"], fontsize=9, fontweight="bold", pad=5)

    parts_list = list(all_results.keys())
    xs = np.arange(len(parts_list)); w = 0.25

    # P1: SS / ROP / EOQ semua part
    ax_p1 = fig_all.add_subplot(gs_all[0, :2]); sax(ax_p1, "Safety Stock | ROP | EOQ – Semua Part")
    ss_all  = [all_results[p]["inv"]["safety_stock"]  for p in parts_list]
    rop_all = [all_results[p]["inv"]["reorder_point"] for p in parts_list]
    eoq_all = [all_results[p]["inv"]["eoq"]           for p in parts_list]
    ax_p1.bar(xs-w, ss_all,  w, label="SS",  color=C["red"],    alpha=0.85)
    ax_p1.bar(xs,   rop_all, w, label="ROP", color=C["orange"], alpha=0.85)
    ax_p1.bar(xs+w, eoq_all, w, label="EOQ", color=C["green"],  alpha=0.85)
    ax_p1.set_xticks(xs); ax_p1.set_xticklabels([p[:14] for p in parts_list], rotation=22, ha="right")
    ax_p1.legend(facecolor=C["bg"], labelcolor=C["text"], fontsize=8)
    ax_p1.set_ylabel("Unit", color=C["text"])

    # P2: Risk Score semua part
    ax_p2 = fig_all.add_subplot(gs_all[0, 2]); sax(ax_p2, "Risk Score – Semua Part")
    risk_scores = [all_results[p]["risk"]["composite_score"] for p in parts_list]
    risk_clrs   = [C["red"] if s>=70 else C["orange"] if s>=45
                   else C["yellow"] if s>=25 else C["green"] for s in risk_scores]
    b_risk = ax_p2.barh([p[:16] for p in parts_list], risk_scores, color=risk_clrs, alpha=0.88)
    for b, v in zip(b_risk, risk_scores):
        ax_p2.text(v+1, b.get_y()+b.get_height()/2, f"{v}", va="center",
                   color=C["text"], fontsize=8, fontweight="bold")
    ax_p2.set_xlim(0, 110); ax_p2.axvline(70, color=C["red"], ls=":", lw=1, alpha=0.5)
    ax_p2.set_xlabel("Score /100", color=C["text"])

    # P3: Demand history overlay semua part
    ax_p3 = fig_all.add_subplot(gs_all[1, :2]); sax(ax_p3, "Demand Historis Semua Part")
    part_colors_cycle = [C["blue"],C["green"],C["orange"],C["purple"],
                         C["teal"],C["yellow"],C["red"],"#FF8A80"]
    for i, pname in enumerate(parts_list[:6]):
        pdata = df[df["part_name"]==pname].sort_values("date")
        ax_p3.plot(pdata["date"].values, pdata["demand"].values,
                   lw=1, alpha=0.75, label=pname[:16],
                   color=part_colors_cycle[i % len(part_colors_cycle)])
    ax_p3.legend(facecolor=C["bg"], labelcolor=C["text"], fontsize=7, ncol=2)
    ax_p3.set_ylabel("Qty", color=C["text"])

    # P4: ABC-XYZ Pie
    ax_p4 = fig_all.add_subplot(gs_all[1, 2]); sax(ax_p4, "ABC-XYZ Distribution")
    abc_xyz: dict[str, int] = {}
    for p, res in all_results.items():
        key = res["inv"]["abc_class"] + res["inv"]["xyz_class"]
        abc_xyz[key] = abc_xyz.get(key, 0) + 1
    pie_colors = [C["red"] if "A" in k else C["orange"] if "B" in k else C["teal"]
                  for k in abc_xyz.keys()]
    ax_p4.pie(list(abc_xyz.values()), labels=list(abc_xyz.keys()),
              colors=pie_colors, autopct="%1.0f%%",
              textprops={"color": C["text"], "fontsize": 9})

    fig_all.text(0.5, 0.97, "NexusGen v2.0 — Dashboard Semua Suku Cadang",
                  ha="center", color=C["text"], fontsize=13, fontweight="bold")
    plt.savefig(r"C:\Users\HP\Downloads\nexusgen_all.png", dpi=120, bbox_inches="tight", facecolor="#0d1420")
    st.pyplot(fig_all, use_container_width=True)
    plt.close()


# ════════════════════════════════════════════════════════════════
# TAB 5: DETAIL INVENTORI
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"## ⚙️ Detail Inventori — {sel_part}")

    d1, d2, d3 = st.columns(3)
    d1.metric("ABC Class",   sel_inv["abc_class"],
              help="A=nilai tinggi, B=menengah, C=rendah")
    d2.metric("XYZ Class",   sel_inv["xyz_class"],
              help="X=stabil, Y=sedang, Z=tidak menentu")
    d3.metric("Service Level", sel_inv["service_level_pct"])

    d4, d5, d6 = st.columns(3)
    d4.metric("Nilai Inventori di ROP", f"Rp {sel_inv['inventory_value_idr']/1e6:.2f}JT")
    d5.metric("Biaya Holding/Tahun",    f"Rp {sel_inv['annual_holding_cost_idr']/1e6:.2f}JT")
    d6.metric("Risiko Stockout",        sel_inv["stockout_risk_pct"])

    st.divider()
    st.markdown("### 📊 Skenario Service Level")
    sl_scenarios = []
    for sl in [90, 95, 97, 99]:
        inv_sc = calc_inventory(
            df[df["part_name"]==sel_part]["demand"].values,
            unit_cost=SPARE_PARTS_CATALOG[sel_part]["unit_cost"],
            criticality=SPARE_PARTS_CATALOG[sel_part]["criticality"],
            service_level=sl/100,
        )
        h = SPARE_PARTS_CATALOG[sel_part]["unit_cost"] * HOLDING_COST_PCT
        sl_scenarios.append({
            "Service Level":    f"{sl}%",
            "Z-Score":          round(inv_sc["Z"],3),
            "Safety Stock":     inv_sc["safety_stock"],
            "ROP":              inv_sc["reorder_point"],
            "EOQ":              inv_sc["eoq"],
            "Hold Cost/Thn":    f"Rp {inv_sc['annual_holding_cost_idr']/1e6:.2f}JT",
            "Stok. Risk":       inv_sc["stockout_risk_pct"],
        })
    st.dataframe(pd.DataFrame(sl_scenarios), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📐 Risk Score Breakdown")
    rb1, rb2, rb3, rb4 = st.columns(4)
    rb1.metric("Composite",     f"{sel_risk['composite_score']}/100", sel_risk["level"])
    rb2.metric("Inventory Gap", f"{sel_risk['inventory_risk']}/100")
    rb3.metric("Demand Uncert.",f"{sel_risk['demand_risk']}/100")
    rb4.metric("Coverage",      f"{sel_risk['coverage_risk']}/100")

    fig_risk, ax_risk = plt.subplots(figsize=(8, 3), facecolor="#0d1420")
    ax_risk.set_facecolor("#080c12")
    cats_r = ["Inventory\nGap", "Demand\nUncert.", "Coverage\nRisk", "COMPOSITE"]
    vals_r = [sel_risk["inventory_risk"], sel_risk["demand_risk"],
              sel_risk["coverage_risk"],  sel_risk["composite_score"]]
    clrs_r = ["#FF5252" if v>=70 else "#FFB74D" if v>=45
               else "#FFF176" if v>=25 else "#69F0AE" for v in vals_r]
    b_r = ax_risk.bar(cats_r, vals_r, color=clrs_r, alpha=0.88)
    for b, v in zip(b_r, vals_r):
        ax_risk.text(b.get_x()+b.get_width()/2, v+1, f"{v}",
                     ha="center", color="#E8EAF6", fontsize=10, fontweight="bold")
    ax_risk.set_ylim(0, 115); ax_risk.axhline(70, color="#FF5252", ls=":", lw=1, alpha=0.5)
    ax_risk.tick_params(colors="#E8EAF6")
    for sp in ax_risk.spines.values(): sp.set_edgecolor("#1e2a3a")
    ax_risk.set_title("Risk Score Breakdown", color="#E8EAF6", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_risk, use_container_width=True)
    plt.close()
