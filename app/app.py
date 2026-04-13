"""
SolarPredict – Streamlit Solar Power Prediction Application
============================================================
Pages  : Dashboard | Manual Prediction | Real-Time Monitor
Storage: predictions.csv (auto-created)
Model  : MLflow registry (falls back to model.pkl)
Weather: OpenWeatherMap (optional – app works without it)
"""

from __future__ import annotations

import random
import time
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    CITY,
    CSV_PATH,
    INTERVAL_OPTIONS,
    PLANT_CAPACITY_KW,
    WEATHER_API_KEY,
    WEATHER_CACHE_TTL,
)
from model_loader import load_model
from utils import create_features, load_csv, save_prediction
from weather_service import get_weather_data

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SolarPredict",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #FFFDF7;
        border: 1px solid #FAEEDA;
        border-radius: 10px;
        padding: 14px 18px;
    }
    /* Primary buttons */
    .stButton > button {
        background: #BA7517 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background: #854F0B !important;
    }
    /* Secondary / outline buttons */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #BA7517 !important;
        border: 1px solid #BA7517 !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFDF7;
    }
    /* Section divider */
    .solar-divider {
        border: none;
        border-top: 1px solid #FAEEDA;
        margin: 1.2rem 0;
    }
    /* Weather card */
    .weather-card {
        background: #FFFDF7;
        border: 1px solid #FAEEDA;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 1rem;
    }
    /* Badge helpers */
    .badge-high   { background:#FAEEDA; color:#633806;
                    padding:3px 10px; border-radius:12px; font-size:13px; }
    .badge-medium { background:#EAF3DE; color:#27500A;
                    padding:3px 10px; border-radius:12px; font-size:13px; }
    .badge-low    { background:#E6F1FB; color:#0C447C;
                    padding:3px 10px; border-radius:12px; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION  (before any page code runs)
# ═══════════════════════════════════════════════════════════════════════════
_defaults: dict = {
    "model":         None,
    "model_loaded":  False,
    "page":          "Dashboard",
    "rt_running":    False,
    "rt_last_run":   None,   # datetime of last auto prediction
    "rt_interval":   3600,   # seconds
    "weather_irr":   None,   # persists live weather into Manual inputs
    "weather_temp":  None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Load model once ────────────────────────────────────────────────────────
if not st.session_state["model_loaded"]:
    with st.spinner("Loading model…"):
        st.session_state["model"] = load_model()
    st.session_state["model_loaded"] = True

# ── Reload CSV every rerun (picks up new rows written by scheduler) ────────
st.session_state["history"] = load_csv(CSV_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# CACHED WEATHER  (defined here so @st.cache_data works correctly)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=WEATHER_CACHE_TTL)
def cached_weather() -> dict | None:
    return get_weather_data(WEATHER_API_KEY, CITY)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED HELPER – auto prediction (used by scheduler + Run Now)
# ═══════════════════════════════════════════════════════════════════════════
def run_auto_prediction() -> tuple[float, str]:
    """
    Determine sensor inputs, run the model, save result.
    Priority: live weather → last CSV row → random fallback.
    Returns (predicted_kw, source_label).
    """
    model = st.session_state["model"]
    df    = st.session_state["history"]

    weather = cached_weather() if WEATHER_API_KEY else None
    if weather:
        sim_irr  = weather["irradiation"]
        sim_temp = weather["temperature"]
        source   = f"live weather ({weather['city']})"
    elif len(df) > 0:
        last     = df.iloc[-1]
        sim_irr  = float(last["irradiation"])
        sim_temp = float(last["amb_temp"])
        source   = "last CSV row"
    else:
        sim_irr  = round(random.uniform(0.3, 0.9), 3)
        sim_temp = round(random.uniform(20.0, 35.0), 1)
        source   = "random fallback"

    sim_dt          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feat_df, inter  = create_features(sim_dt, sim_irr, sim_temp)
    pred            = round(float(model.predict(feat_df)[0]), 2)

    save_prediction(
        CSV_PATH, sim_dt, sim_irr, sim_temp,
        inter["hour_sin"], inter["hour_cos"],
        pred, mode="auto",
    )
    st.session_state["rt_last_run"] = datetime.now()
    return pred, source


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ☀️ SolarPredict")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Dashboard", "Manual Prediction", "Real-Time Monitor"],
        index=["Dashboard", "Manual Prediction", "Real-Time Monitor"]
               .index(st.session_state["page"]),
    )
    st.session_state["page"] = page

    st.markdown("---")

    # Model status
    if st.session_state["model"]:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not loaded")

    # Weather API status
    if WEATHER_API_KEY:
        st.success("✅ Weather API key set")
    else:
        st.warning("⚠️ WEATHER_API_KEY not set\nApp runs without live weather.")

    st.markdown("---")
    st.caption(
        "Run locally:\n"
        "```\nset WEATHER_API_KEY=your_key\n"
        "streamlit run app.py\n```"
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── HELPERS ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _today_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["datetime"].dt.date == date.today()
    return df[mask]


def _output_badge(kw: float) -> str:
    if kw > 3500:
        return '<span class="badge-high">🟠 High</span>'
    if kw > 1500:
        return '<span class="badge-medium">🟡 Medium</span>'
    return '<span class="badge-low">🔵 Low</span>'


def _time_bucket(hour: int) -> str:
    if 5  <= hour < 12: return "🌅 Morning"
    if 12 <= hour < 17: return "☀️ Afternoon"
    if 17 <= hour < 21: return "🌆 Evening"
    return "🌙 Night"


def _irr_label(v: float) -> str:
    if v > 0.7: return "High"
    if v > 0.3: return "Medium"
    return "Low"


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
def page_dashboard() -> None:
    st.title("📊 Dashboard")

    df      = st.session_state["history"]
    today   = _today_df(df)

    # ── Weather widget ──────────────────────────────────────────────────────
    weather = cached_weather()
    st.markdown('<div class="weather-card">', unsafe_allow_html=True)
    w1, w2, w3, w4, w5 = st.columns(5)
    if weather:
        w1.metric("🌡 Temperature",  f"{weather['temperature']} °C")
        w2.metric("💧 Humidity",     f"{weather['humidity']} %")
        w3.metric("☁️ Cloud cover",  f"{weather['clouds_pct']} %")
        w4.metric("☀️ Irradiation",  f"{weather['irradiation']}")
        w5.metric("📍 City",         weather["city"])
        st.caption(
            f"**{weather['description']}**  •  "
            f"Last fetched: {weather['fetched_at']}  •  "
            f"Refreshes every {WEATHER_CACHE_TTL // 60} min"
        )
    else:
        w1.info("Weather unavailable")
        if not WEATHER_API_KEY:
            st.caption("Set WEATHER_API_KEY environment variable to enable live weather.")
        else:
            st.caption("Could not reach OpenWeatherMap — check connection.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    # ── Prediction metrics ──────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    latest_kw = f"{df.iloc[-1]['ac_power']:,.0f} kW" if len(df) > 0 else "—"
    avg_kw    = f"{today['ac_power'].mean():,.1f} kW" if len(today) > 0 else "—"
    sched     = "🟢 Running" if st.session_state["rt_running"] else "🔴 Stopped"
    count_t   = len(today)

    m1.metric("⚡ Latest prediction",   latest_kw)
    m2.metric("📈 Today's average",     avg_kw)
    m3.metric("🔄 Scheduler",           sched)
    m4.metric("🔢 Predictions today",   count_t)

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    # ── Mode navigation cards ───────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 🧮 Manual Prediction")
            st.write(
                "Enter irradiation, temperature, and datetime yourself "
                "to get an instant AC power prediction. Ideal for "
                "what-if analysis and spot checks."
            )
            if st.button("Go to Manual Prediction →", key="nav_manual"):
                st.session_state["page"] = "Manual Prediction"
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("#### ⚡ Real-Time Monitor")
            st.write(
                "Automatically predicts AC power on a schedule using "
                "live weather data. View a live chart and full "
                "prediction history."
            )
            if st.button("Go to Real-Time Monitor →", key="nav_rt"):
                st.session_state["page"] = "Real-Time Monitor"
                st.rerun()

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    # ── Recent predictions table ────────────────────────────────────────────
    st.subheader("Recent Predictions")
    if df.empty:
        st.info("No predictions yet. Make your first prediction above.")
    else:
        display = (
            df[["datetime", "irradiation", "amb_temp", "ac_power", "mode"]]
            .copy()
            .sort_values("datetime", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        display["datetime"] = display["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            display,
            use_container_width=True,
            column_config={
                "datetime":   st.column_config.TextColumn("Date & Time"),
                "irradiation":st.column_config.NumberColumn("Irradiation", format="%.3f"),
                "amb_temp":   st.column_config.NumberColumn("Temp (°C)",   format="%.1f"),
                "ac_power":   st.column_config.ProgressColumn(
                    "AC Power (kW)",
                    min_value=0,
                    max_value=PLANT_CAPACITY_KW,
                    format="%.0f",
                ),
                "mode": st.column_config.TextColumn("Mode"),
            },
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 – MANUAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
def page_manual() -> None:
    st.title("🧮 Manual Prediction")

    model = st.session_state["model"]
    if model is None:
        st.error(
            "❌ Model is not loaded. Make sure model.pkl exists in the app "
            "folder, or that your MLflow server is running."
        )
        return

    col_input, col_features = st.columns(2)

    # ── LEFT: inputs ────────────────────────────────────────────────────────
    with col_input:
        st.markdown("#### Input Parameters")

        datetime_str = st.text_input(
            "Date & Time",
            value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            help="Format: YYYY-MM-DD HH:MM:SS",
        )

        # Live weather button – writes to session_state then reruns
        if st.button("🌦 Use Live Weather", key="live_wx"):
            if not WEATHER_API_KEY:
                st.error("WEATHER_API_KEY environment variable is not set.")
            else:
                w = cached_weather()
                if w:
                    st.session_state["weather_irr"]  = w["irradiation"]
                    st.session_state["weather_temp"] = w["temperature"]
                    st.rerun()
                else:
                    st.error("Failed to fetch weather — check API key or connection.")

        irradiation = st.number_input(
            "Irradiation (W/m²)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(st.session_state["weather_irr"])
                  if st.session_state["weather_irr"] is not None else 0.5,
        )
        amb_temp = st.number_input(
            "Ambient Temperature (°C)",
            min_value=-10.0, max_value=60.0, step=0.1,
            value=float(st.session_state["weather_temp"])
                  if st.session_state["weather_temp"] is not None else 25.0,
        )

        predict_btn = st.button("⚡ Predict AC Power", key="predict_manual")

    # ── RIGHT: live feature preview (updates on every interaction) ──────────
    with col_features:
        st.markdown("#### Computed Features")
        try:
            _, inter = create_features(datetime_str, irradiation, amb_temp)

            fi1, fi2, fi3 = st.columns(3)
            fi1.metric("Hour",     inter["hour"])
            fi2.metric("Hour sin", f"{inter['hour_sin']:.4f}")
            fi3.metric("Hour cos", f"{inter['hour_cos']:.4f}")

            st.markdown("**Feature vector sent to model:**")
            ff1, ff2, ff3, ff4 = st.columns(4)
            ff1.metric("IRRADIATION",    f"{irradiation:.3f}")
            ff2.metric("AMB TEMP",       f"{amb_temp:.1f}")
            ff3.metric("HOUR_SIN",       f"{inter['hour_sin']:.4f}")
            ff4.metric("HOUR_COS",       f"{inter['hour_cos']:.4f}")

        except ValueError as e:
            st.warning(f"⚠️ {e}")

    # ── Prediction output ───────────────────────────────────────────────────
    if predict_btn:
        try:
            feat_df, inter = create_features(datetime_str, irradiation, amb_temp)
        except ValueError as e:
            st.error(f"❌ {e}")
            return

        with st.spinner("Running prediction…"):
            prediction = round(float(model.predict(feat_df)[0]), 2)

        st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        r1, r2 = st.columns([2, 1])
        with r1:
            st.metric(
                "⚡ Predicted AC Power",
                f"{prediction:,.0f} kW",
                delta=f"{(prediction / PLANT_CAPACITY_KW * 100):.1f}% of capacity",
            )
            st.progress(min(prediction / PLANT_CAPACITY_KW, 1.0))

        with r2:
            st.markdown("**Output level**")
            st.markdown(_output_badge(prediction), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("**Conditions**")
            st.markdown(
                f"{_time_bucket(inter['hour'])}  •  "
                f"Irradiation: {_irr_label(irradiation)}"
            )

        # Save & reset live weather overrides
        save_prediction(
            CSV_PATH, datetime_str, irradiation, amb_temp,
            inter["hour_sin"], inter["hour_cos"],
            prediction, mode="manual",
        )
        st.session_state["weather_irr"]  = None
        st.session_state["weather_temp"] = None
        st.success("✅ Prediction saved to predictions.csv")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 – REAL-TIME MONITOR
# ═══════════════════════════════════════════════════════════════════════════
def page_realtime() -> None:
    st.title("⚡ Real-Time Monitor")

    model = st.session_state["model"]
    df    = st.session_state["history"]
    today = _today_df(df)

    # ── Controls ────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)

    with ctrl1:
        sel_interval = st.selectbox(
            "Interval", list(INTERVAL_OPTIONS.keys()),
            index=list(INTERVAL_OPTIONS.values())
                  .index(st.session_state["rt_interval"])
                  if st.session_state["rt_interval"] in INTERVAL_OPTIONS.values()
                  else 1,
        )
        st.session_state["rt_interval"] = INTERVAL_OPTIONS[sel_interval]

    with ctrl2:
        if st.button("▶ Start", key="rt_start"):
            if model is None:
                st.error("Model not loaded — cannot start scheduler.")
            else:
                st.session_state["rt_running"]  = True
                st.session_state["rt_last_run"] = datetime.now()
                st.rerun()

    with ctrl3:
        if st.button("⏹ Stop", key="rt_stop"):
            st.session_state["rt_running"] = False
            st.rerun()

    with ctrl4:
        if st.button("⚡ Run Now", key="rt_runnow"):
            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("Running prediction…"):
                    pred, source = run_auto_prediction()
                st.success(f"✅ {pred:,.0f} kW  (source: {source})")
                st.rerun()

    with ctrl5:
        status_txt = "🟢 Running" if st.session_state["rt_running"] else "🔴 Stopped"
        st.markdown(f"**Status:** {status_txt}")

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    # ── Metrics ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    latest_kw  = f"{df.iloc[-1]['ac_power']:,.0f} kW" if len(df)    > 0 else "—"
    peak_kw    = f"{today['ac_power'].max():,.0f} kW"  if len(today) > 0 else "—"
    avg_kw     = f"{today['ac_power'].mean():,.1f} kW" if len(today) > 0 else "—"
    count_t    = len(today)

    m1.metric("⚡ Latest",        latest_kw)
    m2.metric("📈 Today's peak",  peak_kw)
    m3.metric("📊 Today's avg",   avg_kw)
    m4.metric("🔢 Count today",   count_t)

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    # ── Chart ───────────────────────────────────────────────────────────────
    st.subheader("Predicted AC Power — Today")
    if len(today) > 0:
        chart_df = today.copy().sort_values("datetime")
        fig = px.line(
            chart_df, x="datetime", y="ac_power",
            labels={"ac_power": "AC Power (kW)", "datetime": "Time"},
        )
        fig.add_hline(
            y=PLANT_CAPACITY_KW, line_dash="dash", line_color="#E24B4A",
            annotation_text=f"Capacity ({PLANT_CAPACITY_KW:,} kW)",
            annotation_position="bottom right",
        )
        fig.update_traces(line_color="#BA7517", line_width=2.5)
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#FAEEDA"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet today — press **▶ Start** or **⚡ Run Now**.")

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    # ── Countdown + auto-prediction ─────────────────────────────────────────
    if st.session_state["rt_running"]:
        if model is None:
            st.error("❌ Model not loaded — stopping scheduler.")
            st.session_state["rt_running"] = False
        else:
            now       = datetime.now()
            last_run  = st.session_state["rt_last_run"] or now
            elapsed   = (now - last_run).total_seconds()
            remaining = st.session_state["rt_interval"] - elapsed

            if remaining <= 0:
                with st.spinner("Running scheduled prediction…"):
                    pred, source = run_auto_prediction()
                st.success(f"✅ Auto prediction: {pred:,.0f} kW  (source: {source})")
                st.rerun()
            else:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                src  = "live weather" if WEATHER_API_KEY else "CSV / random"
                st.info(
                    f"⏱ Next auto prediction in **{mins:02d}:{secs:02d}**  •  "
                    f"data source: *{src}*"
                )
                time.sleep(1)
                st.rerun()

    # ── History table ────────────────────────────────────────────────────────
    st.subheader("Today's Prediction History")
    if today.empty:
        st.info("No predictions yet today.")
    else:
        display = (
            today[["datetime", "irradiation", "amb_temp", "ac_power", "mode"]]
            .copy()
            .sort_values("datetime", ascending=False)
            .reset_index(drop=True)
        )
        display["datetime"] = display["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            display,
            use_container_width=True,
            column_config={
                "datetime":    st.column_config.TextColumn("Time"),
                "irradiation": st.column_config.NumberColumn("Irradiation", format="%.3f"),
                "amb_temp":    st.column_config.NumberColumn("Temp (°C)",   format="%.1f"),
                "ac_power":    st.column_config.ProgressColumn(
                    "AC Power (kW)",
                    min_value=0,
                    max_value=PLANT_CAPACITY_KW,
                    format="%.0f",
                ),
                "mode": st.column_config.TextColumn("Mode"),
            },
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════
_page = st.session_state["page"]

if _page == "Dashboard":
    page_dashboard()
elif _page == "Manual Prediction":
    page_manual()
elif _page == "Real-Time Monitor":
    page_realtime()
