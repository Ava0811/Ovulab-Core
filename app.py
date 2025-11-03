# app.py — Ovulab (final, offline, PDF-enabled)
import os
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.units import cm
from datetime import datetime
import pytz

# ---------------------- Page & Theme ----------------------
st.set_page_config(page_title="Ovulab | Ovulation Irregularity Assistant",
                   page_icon="💜", layout="centered")

# Lavender wellness theme
PRIMARY = "#8C73C9"     # lavender-purple accent
PRIMARY_SOFT = "#EEE9FA"
TEXT_DARK = "#2F2A3B"
CARD_BG = "#FFFFFF"
BORDER = "#E8E2F6"

st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    color: {TEXT_DARK};
}}
.header {{
    background: linear-gradient(135deg, {PRIMARY_SOFT} 0%, #F7F5FD 100%);
    padding: 24px 28px;
    border-radius: 16px;
    border: 1px solid {BORDER};
    margin-bottom: 18px;
}}
.brand {{ display: flex; align-items: center; gap: 12px; }}
.card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 18px;
    margin: 10px 0;
}}
.stButton>button {{
    background: {PRIMARY}; color: white; border: 0; border-radius: 10px;
    padding: 10px 14px; font-weight: 600;
}}
.stButton>button:hover {{ background: #7A64B2; }}
.pill {{
    display: inline-block; padding: 6px 10px; border-radius: 999px;
    font-weight: 700; font-size: 12px; border: 1px solid {BORDER};
    background: {PRIMARY_SOFT}; color: {TEXT_DARK};
}}
.success {{
    background: #F2EFFC; border: 1px solid {BORDER}; border-left: 6px solid {PRIMARY};
    padding: 12px 14px; border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# Logo (soft, Style B)
st.markdown(f"""
<div class="header">
  <div class="brand">
    <div style="
      background: {PRIMARY_SOFT};
      color: {PRIMARY};
      border: 1px solid {BORDER};
      border-radius: 14px;
      padding: 6px 12px;
      font-weight: 700;
      font-size: 14px;">
      Ovulab
    </div>
    <div>
      <div style="margin:0; font-weight:800; font-size:26px; color:{TEXT_DARK};">Cycle-aware Nutrition & Balance</div>
      <div class="sub" style="color:#5E5678; font-size:14px; margin-top:6px;">
        Gentle, science-backed guidance for ovulation regularity.
      </div>
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
    return {"Light": 2.0, "Moderate": 5.0, "Heavy": 8.0}.get(label, 5.0)

def risk_level_3(cycle_length: float, cycle_var: float) -> int:
    """
    3-level rules:
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
    """Policy: if RF predicts irregular (1), upgrade rule level by +1 (cap at 2)."""
    if rf_pred == 1 and base_level < 2:
        return base_level + 1
    return base_level

def level_to_text(level: int) -> str:
    return ["LOW", "MEDIUM", "HIGH"][int(level)]

def level_to_color(level: int) -> str:
    return ["#34A853", "#FBBC05", "#EA4335"][int(level)]  # green, amber, red

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
st.markdown("### 🔮 Predict & Build Plan")

go = st.button("Predict Risk & Build Plan")
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

    # ---------- Data tables (no API) ----------
    def build_nutrition_df(level_text: str) -> pd.DataFrame:
        base = {
            "LOW": [
                {"Nutrient Focus":"Fiber", "Why (Mechanism)":"Glucose–insulin stability → protects LH sensitivity",
                 "Foods (India-dominant)":"Oats, millets, vegetables, fruit w/ peel", "Frequency":"Every meal"},
                {"Nutrient Focus":"Omega-3", "Why (Mechanism)":"Anti-inflammatory follicular environment",
                 "Foods (India-dominant)":"Flaxseed, chia, walnuts", "Frequency":"4–5 days/week"},
                {"Nutrient Focus":"Vitamin D", "Why (Mechanism)":"Steroidogenesis & luteal progesterone signaling",
                 "Foods (India-dominant)":"Sunlight (7–10am), fortified milk/curd", "Frequency":"Daily"},
                {"Nutrient Focus":"Magnesium", "Why (Mechanism)":"LH receptor support; lowers luteal stress response",
                 "Foods (India-dominant)":"Spinach/methi, seeds, almonds", "Frequency":"Daily"},
                {"Nutrient Focus":"Protein", "Why (Mechanism)":"Amino acids for hormone synthesis & satiety",
                 "Foods (India-dominant)":"Paneer, curd, lentils", "Frequency":"Each major meal"},
            ],
            "MEDIUM": [
                {"Nutrient Focus":"Low-GI Fibers", "Why (Mechanism)":"Lower insulin peaks → ↓ ovarian androgen drive",
                 "Foods (India-dominant)":"Bajra, jowar, rajma, kala chana", "Frequency":"Each meal"},
                {"Nutrient Focus":"Polyphenols", "Why (Mechanism)":"Lower inflammation disrupting ovulation timing",
                 "Foods (India-dominant)":"Amla, green tea, berries (or jamun)", "Frequency":"3–4 days/week"},
                {"Nutrient Focus":"Protein front-loading", "Why (Mechanism)":"Stabilizes morning glucose curve",
                 "Foods (India-dominant)":"Paneer+veg or eggs+veg breakfast", "Frequency":"Daily"},
                {"Nutrient Focus":"Turmeric + Pepper", "Why (Mechanism)":"Curcumin dampens inflammatory enzymes",
                 "Foods (India-dominant)":"Haldi milk / turmeric seasoning", "Frequency":"Daily"},
                {"Nutrient Focus":"Magnesium", "Why (Mechanism)":"Supports luteal hormone response",
                 "Foods (India-dominant)":"Seeds + greens combo", "Frequency":"Daily"},
            ],
            "HIGH": [
                {"Nutrient Focus":"Zero refined sugar phase", "Why (Mechanism)":"↓ insulin surge → ↓ ovarian androgen suppression",
                 "Foods (India-dominant)":"No bakery/sweets/packaged", "Frequency":"Strict 2 weeks"},
                {"Nutrient Focus":"Fiber-first method", "Why (Mechanism)":"Slows carb absorption → stable glucose",
                 "Foods (India-dominant)":"Raw veg + protein BEFORE roti/rice", "Frequency":"Every main meal"},
                {"Nutrient Focus":"Omega-3", "Why (Mechanism)":"Anti-inflammatory follicular fluid",
                 "Foods (India-dominant)":"Flax + chia (1 tbsp each)", "Frequency":"Daily"},
                {"Nutrient Focus":"Iron + Folate", "Why (Mechanism)":"Supports bleeding quality & recovery",
                 "Foods (India-dominant)":"Palak, beetroot, chole, rajma", "Frequency":"3–4 days/week"},
                {"Nutrient Focus":"Sleep nutrition", "Why (Mechanism)":"Melatonin rhythm → ovulatory timing",
                 "Foods (India-dominant)":"No caffeine after 5pm; light dinner", "Frequency":"Daily"},
            ],
        }
        return pd.DataFrame(base[level_text])

    def build_mind_df(level_text: str) -> pd.DataFrame:
        base = {
            "LOW": [
                {"Pillar":"Exercise","Prescription":"30 min brisk walk + 15 min yoga stretch","Frequency":"5–6 days/week"},
                {"Pillar":"Meditation","Prescription":"Box-breathing (4-4-4-4) at night","Frequency":"5–10 min daily"},
            ],
            "MEDIUM": [
                {"Pillar":"Exercise","Prescription":"40 min brisk walk + 2×/week light dumbbells","Frequency":"5 days/week"},
                {"Pillar":"Meditation","Prescription":"Yoga nidra or mindful belly breathing","Frequency":"15 min daily"},
            ],
            "HIGH": [
                {"Pillar":"Exercise","Prescription":"45 min walk + 3×/week full-body resistance","Frequency":"6 days/week"},
                {"Pillar":"Meditation","Prescription":"Guided meditation + 5 min journaling","Frequency":"20 min daily"},
            ],
        }
        return pd.DataFrame(base[level_text])

    def build_mealplan_df(level_text: str) -> pd.DataFrame:
        base = {
            "LOW": [
                {"Day":"Day 1","Plan":"Oats + curd + fruit; Dal-chawal + sabzi; Light dinner + haldi milk"},
                {"Day":"Day 2","Plan":"Veg poha + seeds; Roti + paneer sabzi; Tomato soup + salad"},
                {"Day":"Day 3","Plan":"Upma + nuts; Rajma + rice + salad; Khichdi + curd"},
                {"Day":"Day 4","Plan":"Idli + sambar; Millet roti + veg; Dalia + sprouts"},
                {"Day":"Day 5","Plan":"Besan chilla; Chole + rice; Veg stew + roti"},
                {"Day":"Day 6","Plan":"Paratha (little ghee) + curd; Fish/Tofu + veg; Khichdi + dahi"},
                {"Day":"Day 7","Plan":"Fruit + seeds bowl; Dal + roti + sabzi; Soup + salad"},
            ],
            "MEDIUM": [
                {"Day":"Day 1","Plan":"Ragi dosa; Bajra roti + paneer; Dal + salad"},
                {"Day":"Day 2","Plan":"Paneer bhurji + millets; Rajma + rice + kachumber; Veg soup"},
                {"Day":"Day 3","Plan":"Oats + seeds; Kala chana + roti; Daliya + curd"},
                {"Day":"Day 4","Plan":"Eggs/Tofu + veg; Jowar roti + sabzi; Lemon rasam + veggie"},
                {"Day":"Day 5","Plan":"Poha + peanuts; Palak paneer + rice; Moong soup"},
                {"Day":"Day 6","Plan":"Idli + sambar; Chole + roti; Quinoa bowl"},
                {"Day":"Day 7","Plan":"Masala oats; Dal + sabzi + salad; Khichdi + dahi"},
            ],
            "HIGH": [
                {"Day":"Day 1","Plan":"Eggs/Paneer + veg (no bread); Chole + salad then rice; Soup + veg"},
                {"Day":"Day 2","Plan":"Besan chilla + curd; Rajma + salad then rice; Veg + roti late"},
                {"Day":"Day 3","Plan":"Tofu/paneer bowl; Palak + roti (late); Khichdi + salad"},
                {"Day":"Day 4","Plan":"Veg omelette/paneer; Bajra roti + sabzi; Dal + veg"},
                {"Day":"Day 5","Plan":"Sprouts + fruit; Fish/Tofu + veg; Daliya + curd"},
                {"Day":"Day 6","Plan":"Ragi dosa + sambar; Chana + salad then roti; Veg stew"},
                {"Day":"Day 7","Plan":"Oats + seeds; Mixed daal + salad; Soup + salad"},
            ],
        }
        return pd.DataFrame(base[level_text])

    # Build & show tables
    level_text = level_to_text(final_level)

    st.markdown("### 🍽️ Nutrition — Mechanism-based Targets")
    nutri_df = build_nutrition_df(level_text)
    st.dataframe(nutri_df, use_container_width=True)
    st.download_button("Download nutrition table (CSV)",
                       data=nutri_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"ovulab_nutrition_{level_text.lower()}.csv", mime="text/csv")

    st.markdown("### 🧘‍♀️ Exercise & Meditation — Prescription")
    mind_df = build_mind_df(level_text)
    st.dataframe(mind_df, use_container_width=True)
    st.download_button("Download exercise/meditation table (CSV)",
                       data=mind_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"ovulab_mind_{level_text.lower()}.csv", mime="text/csv")

    st.markdown("### 🗓️ 7-Day Sample Meal Plan")
    meal_df = build_mealplan_df(level_text)
    st.dataframe(meal_df, use_container_width=True)
    st.download_button("Download 7-day plan (CSV)",
                       data=meal_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"ovulab_7day_{level_text.lower()}.csv", mime="text/csv")

    # ---------- PDF (Style C, Portrait, IST timestamp bottom-left with weekday) ----------
    def df_to_table_data(df: pd.DataFrame):
        header = list(df.columns)
        rows = df.values.tolist()
        return [header] + rows

    def build_pdf(level_text: str, nutri_df: pd.DataFrame, mind_df: pd.DataFrame, meal_df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=28, bottomMargin=28)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(name="Title", parent=styles["Heading1"],
                                     fontSize=18, textColor=colors.HexColor(PRIMARY), spaceAfter=8)
        subtitle_style = ParagraphStyle(name="Sub", parent=styles["Normal"],
                                        fontSize=11, textColor=colors.HexColor("#5E5678"))
        section_style = ParagraphStyle(name="Section", parent=styles["Heading2"],
                                       fontSize=14, textColor=colors.HexColor(TEXT_DARK), spaceBefore=10, spaceAfter=6)
        risk_style = ParagraphStyle(name="Risk", parent=styles["Normal"],
                                    fontSize=12, textColor=colors.HexColor(TEXT_DARK))

        elements = []
        # Header
        elements.append(Paragraph("Ovulab", title_style))
        elements.append(Paragraph("Modern science for hormonal resilience", subtitle_style))  # Subtitle ii
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Risk level: <b>{level_text}</b>", risk_style))
        elements.append(Spacer(1, 10))

        # Tables builder
        def table_block(title: str, data_df: pd.DataFrame):
            elements.append(Paragraph(title, section_style))
            data = df_to_table_data(data_df)
            tbl = Table(data, repeatRows=1, hAlign='LEFT', colWidths='*')
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor(PRIMARY_SOFT)),
                ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor(TEXT_DARK)),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 10),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor(BORDER)),
                ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
                ('FONTSIZE', (0,1), (-1,-1), 9),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            elements.append(tbl)
            elements.append(Spacer(1, 10))

        # Add sections
        table_block("Nutrition — Mechanism-based Targets", nutri_df)
        table_block("Exercise & Meditation — Prescription", mind_df)
        table_block("7-Day Sample Meal Plan", meal_df)

        # Footer timestamp (bottom-left, IST, with weekday)
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        footer = now_ist.strftime("Ovulab • Generated %a, %d-%b-%Y, %H:%M IST")
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(footer, styles["Normal"]))

        doc.build(elements)
        pdf = buf.getvalue()
        buf.close()
        return pdf

    pdf_bytes = build_pdf(level_text, nutri_df, mind_df, meal_df)
    st.download_button(
        label="📄 Download Full PDF Report",
        data=pdf_bytes,
        file_name=f"ovulab_report_{level_text.lower()}.pdf",
        mime="application/pdf"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Evaluation Card ----------------------
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
            ax.imshow(cm, cmap="Purples")
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
        st.error(f"Could not evaluate: {repr(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Footer ----------------------
st.caption("Disclaimer: This app provides general wellness information and is not a substitute for professional medical advice.")
