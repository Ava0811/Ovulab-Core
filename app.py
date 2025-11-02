# ==============================
# 🔎 Evaluation on a test CSV
# ==============================
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from openai import OpenAI

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------- Page & Theme ----------------------
st.set_page_config(page_title="Ovulab | Ovulation Irregularity Assistant",
                   page_icon="💜", layout="centered")

# Minimal lavender wellness theme via CSS
PRIMARY = "#8C73C9"   # lavender-purple accent (more purple)
PRIMARY_SOFT = "#EEE9FA"
TEXT_DARK = "#2F2A3B"
CARD_BG = "#FFFFFF"
BORDER = "#E8E2F6"

st.markdown(f"""
<style>
/* Global */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
    color: {TEXT_DARK};
}}
/* Header */
.header {{
    background: linear-gradient(135deg, {PRIMARY_SOFT} 0%, #F7F5FD 100%);
    padding: 24px 28px;
    border-radius: 16px;
    border: 1px solid {BORDER};
    margin-bottom: 18px;
}}
.brand {{
    display: flex; align-items: center; gap: 12px;
}}
.brand-badge {{
    background: {PRIMARY};
    color: white;
    border-radius: 12px;
    padding: 6px 10px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: .3px;
}}
.h1 {{
    margin: 0;
    font-weight: 800; font-size: 28px; color: {TEXT_DARK};
}}
.sub {{
    margin-top: 6px; color: #5E5678; font-size: 14px;
}}
/* Cards */
.card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 18px;
    margin: 10px 0;
}}
.card h3 {{
    font-size: 18px; margin: 0 0 10px 0; color: {TEXT_DARK}; font-weight: 700;
}}
/* Buttons */
.stButton>button {{
    background: {PRIMARY};
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight: 600;
}}
.stButton>button:hover {{
    background: #7A64B2;
}}
/* Pills */
.pill {{
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12px;
    border: 1px solid {BORDER};
    background: {PRIMARY_SOFT};
    color: {TEXT_DARK};
}}
/* Success panel */
.success {{
    background: #F2EFFC;
    border: 1px solid {BORDER};
    border-left: 6px solid {PRIMARY};
    padding: 12px 14px; border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header">
  <div class="brand">
    <div class="brand-badge">OVULAB</div>
    <div>
      <div class="h1">Ovulation Irregularity Assistant</div>
      <div class="sub">Gentle, science-backed guidance for cycle regularity and hormone balance.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------- Load Assets ----------------------
@st.cache_resource
def load_assets():
    rf = joblib.load("ovulation_rf_model.pkl")
    scaler = joblib.load("ovulation_scaler.pkl")
    return rf, scaler

try:
    rf, scaler = load_assets()
except Exception as e:
    st.error(f"Could not load model/scaler: {e}")
    st.stop()

# ---------------------- Helpers ----------------------
def bleeding_volume_map(label: str) -> float:
    mapping = {"Light": 2.0, "Moderate": 5.0, "Heavy": 8.0}
    return mapping.get(label, 5.0)

def risk_level_3(cycle_length: float, cycle_var: float) -> int:
    """
    Clinical rule:
    - LOW (0): 26–32 days AND variation < 3
    - MED (1): 22–25 or 33–35 OR variation 3–7
    - HIGH (2): <22 or >35 OR variation > 7
    """
    if (26 <= cycle_length <= 32) and (cycle_var < 3):
        return 0
    elif ((22 <= cycle_length <= 25) or (33 <= cycle_length <= 35)) or (3 <= cycle_var <= 7):
        return 1
    else:
        return 2

def combine_rule_rf(base_level: int, rf_pred: int) -> int:
    """Policy B: if RF predicts irregular (1), upgrade rule level by +1 (cap at 2)."""
    if rf_pred == 1 and base_level < 2:
        return base_level + 1
    return base_level

def level_to_text(level: int) -> str:
    return ["LOW", "MEDIUM", "HIGH"][int(level)]

def level_to_color(level: int) -> str:
    return ["#34A853", "#FBBC05", "#EA4335"][int(level)]  # green, amber, red

# ---------------------- Sidebar: API key ----------------------
with st.sidebar:
    st.markdown("### 🔐 OpenAI API")
    st.caption("Set your OpenAI API key here (or keep via env var).")
    api_key_input = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    st.markdown("---")
    st.caption("Files expected next to app.py:\n• ovulation_rf_model.pkl\n• ovulation_scaler.pkl")

# ---------------------- Inputs Card ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### ✍️ Enter recent cycle details")

c1, c2, c3 = st.columns(3)
last1 = c1.number_input("Last cycle length (days)", min_value=10, max_value=90, value=28, step=1)
last2 = c2.number_input("Cycle length before that (days)", min_value=10, max_value=90, value=29, step=1)
last3 = c3.number_input("Cycle length before that (days)", min_value=10, max_value=90, value=27, step=1)

c4, c5 = st.columns([1,1])
avg_bleeding_days = c4.number_input("Average bleeding days", min_value=1.0, max_value=12.0, value=5.0, step=0.5)
bleed_label = c5.selectbox("Bleeding volume", ["Light", "Moderate", "Heavy"], index=1)

# Derived
cycle_lengths = np.array([last1, last2, last3], dtype=float)
cycle_length_mean = float(np.mean(cycle_lengths))
cycle_length_std = float(np.std(cycle_lengths, ddof=1))
bleeding_volume_score = float(bleeding_volume_map(bleed_label))

st.caption("Computed from your entries:")
st.markdown(
    f"""
    <span class="pill">Mean cycle length: <b>{cycle_length_mean:.1f} d</b></span>
    &nbsp;&nbsp;<span class="pill">Variation (std dev): <b>{cycle_length_std:.2f} d</b></span>
    &nbsp;&nbsp;<span class="pill">Bleeding days: <b>{avg_bleeding_days:.1f} d</b></span>
    &nbsp;&nbsp;<span class="pill">Volume score: <b>{bleeding_volume_score:.1f}</b> ({bleed_label})</span>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Predict Card ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🔮 Predict & Combine")

go = st.button("Predict Risk & Generate Plan")
if go:
    input_df = pd.DataFrame([{
        "cycle_length": cycle_length_mean,
        "cycle_length_variation": cycle_length_std,
        "avg_bleeding_days": avg_bleeding_days,
        "bleeding_volume_score": bleeding_volume_score
    }])

    X_scaled = scaler.transform(input_df)
    rf_pred = int(rf.predict(X_scaled)[0])
    rf_prob = float(rf.predict_proba(X_scaled)[0][1]) if hasattr(rf, "predict_proba") else None

    base_level = risk_level_3(cycle_length_mean, cycle_length_std)
    final_level = combine_rule_rf(base_level, rf_pred)

    color = level_to_color(final_level)
    st.markdown(
        f"""
        <div class="success">
          <b>RF (binary):</b> {rf_pred} {'(Irregular)' if rf_pred==1 else '(Regular)'}<br>
          {'<b>RF probability irregular:</b> ' + str(round(rf_prob,3)) if rf_prob is not None else ''}
          <br><b>Rule level:</b> {level_to_text(base_level)}<br>
          <span class="pill" style="border-color:{color}; background:#FAF8FF;">
            Final 3-level risk: <span style="color:{color}"><b>{level_to_text(final_level)}</b></span>
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------- LLM recommendations ----------
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OpenAI API key not set. Enter it in the sidebar to get recommendations.")
    else:
        client = OpenAI()
        system_msg = (
            "You are a clinical nutrition assistant for menstrual health and PCOS. "
            "Provide practical, food-first recommendations that support ovulation regularity, "
            "insulin sensitivity, inflammation reduction, and hormone balance. Be concise and actionable."
        )
        user_msg = f"""
Risk level: {level_to_text(final_level)}
Inputs:
- cycle_length_mean_days: {cycle_length_mean:.1f}
- cycle_length_variation_days: {cycle_length_std:.2f}
- avg_bleeding_days: {avg_bleeding_days:.1f}
- bleeding_volume_score: {bleeding_volume_score:.1f} (Light≈2, Moderate≈5, Heavy≈8)

Task:
1) In 5 focused bullets, recommend nutrition strategies (Vitamin D, Omega-3, Magnesium, Zinc, Fiber/Low-GI).
2) For each bullet, list 3–4 common Indian foods.
3) Add 2 lifestyle actions (sleep, exercise/sunlight) tuned to the risk.
4) End with a 7-day simple meal outline (1 line/day), vegetarian-friendly options okay.
Avoid medical claims; this is general advice, not a diagnosis.
"""

        with st.spinner("Building your nutrition & lifestyle plan…"):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.7,
                )
                advice = resp.choices[0].message.content
                st.markdown("### 🍽️ Nutrition & Lifestyle Plan")
                st.write(advice)
            except Exception as e:
                st.error(f"OpenAI error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Evaluation Card ----------------------
import pandas as pd  
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🔎 Evaluate on a Test CSV")
st.caption("Upload CSV with columns: cycle_length, cycle_length_variation, avg_bleeding_days, bleeding_volume_score, duration_abnormality_flag")

file = st.file_uploader("Upload CSV to see metrics & confusion matrix", type=["csv"])
if file is not None:
    try:
        test_df = pd.read_csv(file)
        test_df.columns = test_df.columns.str.strip().str.lower()
        required_cols = ["cycle_length","cycle_length_variation","avg_bleeding_days","bleeding_volume_score","duration_abnormality_flag"]
        missing = [c for c in required_cols if c not in test_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X_eval = test_df[["cycle_length","cycle_length_variation","avg_bleeding_days","bleeding_volume_score"]].copy()
            y_true = test_df["duration_abnormality_flag"].astype(int).values

            X_eval_scaled = scaler.transform(X_eval)
            y_pred_bin = rf.predict(X_eval_scaled)

            # Also show final 3-level distribution (rule + RF)
            rule_levels = [
                risk_level_3(row["cycle_length"], row["cycle_length_variation"])
                for _, row in X_eval.iterrows()
            ]
            final_levels = [combine_rule_rf(rule, pred) for rule, pred in zip(rule_levels, y_pred_bin)]

            st.markdown("#### Binary performance (0=regular, 1=irregular)")
            report_txt = classification_report(y_true, y_pred_bin, digits=3)
            st.code(report_txt, language="text")

            cm = confusion_matrix(y_true, y_pred_bin, labels=[0,1])
            fig, ax = plt.subplots(figsize=(4,3))
            im = ax.imshow(cm, cmap="Purples")
            ax.set_title("Confusion Matrix (Binary)")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            st.pyplot(fig)

            st.markdown("#### Final 3-level risk distribution (after combining rule + RF)")
            vals, counts = np.unique(final_levels, return_counts=True)
            st.write({level_to_text(v): int(c) for v, c in zip(vals, counts)})

    except Exception as e:
        st.error(f"Could not evaluate: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Footer ----------------------
st.caption("Disclaimer: This app provides general wellness information and is not a substitute for professional medical advice.") 
This is the exact code 


