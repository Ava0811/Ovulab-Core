# app.py — Ovulab (Final UX + ML + Hybrid System)

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import google.generativeai as genai
from datetime import date

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Ovulab", page_icon="💜", layout="centered")

# ---------------------- UI STYLES ----------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg, #F7F5FD, #EEE9FA);}

.header {
    background: linear-gradient(135deg, #8C73C9, #B6A5E8);
    color: white;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 8px 20px rgba(140,115,201,0.2);
}

.card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    margin: 12px 0;
    border: 1px solid #E8E2F6;
    box-shadow: 0px 6px 18px rgba(140,115,201,0.08);
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
}

.card-title {
    font-size: 17px;
    font-weight: 700;
    color: #8C73C9;
}

.card-content {
    font-size: 14px;
    color: #3F3A52;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("""
<div class="header">
<h2>💜 Ovulab</h2>
<p>Cycle-aware Nutrition & Hormonal Balance</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- CARD FUNCTION ----------------------
def show_card(title, content):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_assets():
    return joblib.load("ovulation_rf_model.pkl"), joblib.load("ovulation_scaler.pkl")

rf, scaler = load_assets()

# ---------------------- HELPERS ----------------------
def bleeding_volume_map(label):
    return {"Light":2, "Moderate":5, "Heavy":8}[label]

def risk_level_3(cycle_length, var):
    if 26 <= cycle_length <= 32 and var < 3:
        return 0
    elif (22 <= cycle_length <= 25 or 33 <= cycle_length <= 35) or (3 <= var <= 7):
        return 1
    return 2

def combine_rule_rf(rule, rf_pred):
    return min(rule + rf_pred, 2)

def level_to_text(level):
    return ["LOW","MEDIUM","HIGH"][level]

def infer_hormone_state(cycle_length, variation):
    if cycle_length > 35:
        return "Delayed Ovulation Pattern"
    elif cycle_length < 22:
        return "Luteal Phase Insufficiency Pattern"
    elif variation > 7:
        return "Hormonal Variability Pattern"
    return "Stable Ovulatory Pattern"

# ---------------------- INPUT MODE ----------------------
mode = st.radio("Choose input method:", ["Enter Cycle Lengths", "Enter Dates (Auto Calculate)"])

# ---------------------- INPUTS ----------------------
if mode == "Enter Cycle Lengths":
    st.caption("Cycle length = days between the start of one period and the next")

    l1 = st.number_input("Last cycle length", 10, 90, 28, help="Days between last two periods")
    l2 = st.number_input("Previous cycle length", 10, 90, 29)
    l3 = st.number_input("Earlier cycle length", 10, 90, 27)

else:
    st.caption("Select first day of your last 3 periods")

    d1 = st.date_input("Most recent period start", value=date(2024,1,1))
    d2 = st.date_input("Previous period start", value=date(2023,12,1))
    d3 = st.date_input("Earlier period start", value=date(2023,11,1))

    l1 = abs((d1 - d2).days)
    l2 = abs((d2 - d3).days)
    l3 = l2  # fallback

    st.success(f"Calculated cycle lengths: {l1}, {l2}, {l3}")

# Bleeding input
bleed = st.number_input(
    "Bleeding duration (days)",
    1.0, 12.0, 5.0,
    help="Number of days your period lasts (not cycle length)"
)

volume = st.selectbox("Bleeding volume", ["Light","Moderate","Heavy"])

# ---------------------- PREDICT ----------------------
if st.button("Predict"):

    lengths = np.array([l1,l2,l3])
    mean = np.mean(lengths)
    var = np.std(lengths, ddof=1)

    input_df = pd.DataFrame([{
        "cycle_length": mean,
        "cycle_length_variation": var,
        "avg_bleeding_days": bleed,
        "bleeding_volume_score": bleeding_volume_map(volume)
    }])

    X = scaler.transform(input_df)
    rf_pred = int(rf.predict(X)[0])

    rule = risk_level_3(mean, var)
    final = combine_rule_rf(rule, rf_pred)

    hormone = infer_hormone_state(mean, var)

    show_card("🧬 Hormone Insight", hormone)

    show_card("📊 Cycle Metrics",
              f"Mean: {mean:.1f} days | Variation: {var:.2f}")

    show_card("⚠️ Risk Level", level_to_text(final))

    # ---------------------- PLAN ----------------------
    plan = f"""
Your cycle pattern suggests **{hormone}**.

### Key actions:
• Maintain regular sleep cycle  
• Balanced meals with protein  
• Daily movement (30–45 min)  
• Reduce refined sugar  

### Sample Plan:
Day 1: Dal + roti + sabzi  
Day 2: Millet bowl + paneer  
Day 3: Rice + chole  
"""

    show_card("📋 Core Plan", plan)

    # ---------------------- GEMINI ----------------------
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
            Hormone: {hormone}
            Risk: {level_to_text(final)}

            Expand into lifestyle + meal plan.
            """

            response = model.generate_content(prompt)

            show_card("🤖 AI Plan", response.text)

        except:
            st.warning("AI unavailable")

    else:
        st.info("Add GEMINI_API_KEY for AI plan")

# ---------------------- FOOTER ----------------------
st.caption("This is general guidance and not a medical diagnosis.")
