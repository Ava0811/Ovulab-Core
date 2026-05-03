# app.py — Ovulab (Hybrid ML + Rule-based + Optional LLM)

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import google.generativeai as genai

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="Ovulab", page_icon="💜", layout="centered")

PRIMARY = "#8C73C9"
PRIMARY_SOFT = "#EEE9FA"

# ---------------------- Header ----------------------
st.markdown(f"""
<div style="background:{PRIMARY_SOFT}; padding:20px; border-radius:15px;">
<h2 style="color:{PRIMARY}; margin:0;">Ovulab</h2>
<p>Cycle-aware Nutrition & Hormonal Balance</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_assets():
    rf = joblib.load("ovulation_rf_model.pkl")
    scaler = joblib.load("ovulation_scaler.pkl")
    return rf, scaler

rf, scaler = load_assets()

# ---------------------- Helper Functions ----------------------
def bleeding_volume_map(label):
    return {"Light":2, "Moderate":5, "Heavy":8}[label]

def risk_level_3(cycle_length, var):
    if 26 <= cycle_length <= 32 and var < 3:
        return 0
    elif (22 <= cycle_length <= 25 or 33 <= cycle_length <= 35) or (3 <= var <= 7):
        return 1
    else:
        return 2

def combine_rule_rf(rule, rf_pred):
    return min(rule + rf_pred, 2)

def level_to_text(level):
    return ["LOW","MEDIUM","HIGH"][level]

# ---------------------- Hormone Inference ----------------------
def infer_hormone_state(cycle_length, variation):
    if cycle_length > 35:
        return "Delayed Ovulation Pattern"
    elif cycle_length < 22:
        return "Luteal Phase Insufficiency Pattern"
    elif variation > 7:
        return "Hormonal Variability Pattern"
    else:
        return "Stable Ovulatory Pattern"

# ---------------------- Nutrition Mapping ----------------------
hormone_nutrition_map = {
    "Delayed Ovulation Pattern": {
        "focus": "Improve insulin sensitivity and estrogen balance",
        "nutrients": ["Omega-3","Fiber","Magnesium"],
        "foods": ["Flax seeds","Chia seeds","Green vegetables","Millets"]
    },
    "Luteal Phase Insufficiency Pattern": {
        "focus": "Support progesterone stability",
        "nutrients": ["Vitamin B6","Magnesium","Zinc"],
        "foods": ["Banana","Nuts","Seeds","Whole grains"]
    },
    "Hormonal Variability Pattern": {
        "focus": "Reduce inflammation and stabilize hormones",
        "nutrients": ["Omega-3","Vitamin D","Antioxidants"],
        "foods": ["Walnuts","Amla","Leafy greens","Sunlight exposure"]
    },
    "Stable Ovulatory Pattern": {
        "focus": "Maintain hormonal balance",
        "nutrients": ["Balanced micronutrients"],
        "foods": ["Fruits","Vegetables","Proteins"]
    }
}

micronutrient_map = {
    "Vitamin D": "Sunlight 15–20 min, fortified milk",
    "Folate": "Spinach, beetroot, legumes",
    "Magnesium": "Pumpkin seeds, almonds",
    "Zinc": "Seeds, nuts"
}

# ---------------------- Rule-Based Plan Generator ----------------------
def generate_plan(hormone_state, data):
    return f"""
### 🧠 Explanation  
Your cycle pattern suggests **{hormone_state}**. This may indicate variations in hormonal rhythm, which can be supported through nutrition and lifestyle adjustments.

### 🥗 Nutrition Strategy  
Focus on: **{data['focus']}**

### 🍽️ Recommended Foods  
{', '.join(data['foods'])}

### 🧪 Key Nutrients  
{', '.join(data['nutrients'])}

### 🧘 Lifestyle Routine  
- Maintain consistent sleep cycle (7–8 hrs)  
- 30–45 min daily movement (walk/yoga)  
- Reduce processed sugar intake  
- Morning sunlight exposure  

### 🗓️ 3-Day Sample Plan  
Day 1: Dal + roti + sabzi + salad  
Day 2: Millet bowl + paneer + greens  
Day 3: Rice + chole + vegetable curry  
"""

# ---------------------- Inputs ----------------------
st.markdown("### Enter Cycle Details")

c1, c2, c3 = st.columns(3)
l1 = c1.number_input("Cycle 1", 10, 90, 28)
l2 = c2.number_input("Cycle 2", 10, 90, 29)
l3 = c3.number_input("Cycle 3", 10, 90, 27)

bleed = st.number_input("Bleeding days", 1.0, 12.0, 5.0)
volume = st.selectbox("Bleeding volume", ["Light","Moderate","Heavy"])

# ---------------------- Predict ----------------------
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

    X_scaled = scaler.transform(input_df)
    rf_pred = int(rf.predict(X_scaled)[0])

    rule = risk_level_3(mean, var)
    final = combine_rule_rf(rule, rf_pred)

    st.success(f"Risk Level: {level_to_text(final)}")

    # Hormone Layer
    hormone_state = infer_hormone_state(mean, var)
    data = hormone_nutrition_map[hormone_state]

    st.markdown("### 🧬 Hormone Insight")
    st.write(hormone_state)

    st.markdown("### 🥗 Nutrition Focus")
    st.write(data["focus"])

    st.markdown("### 🥦 Foods")
    st.write(", ".join(data["foods"]))

    st.markdown("### 🧪 Micronutrients")
    for k,v in micronutrient_map.items():
        st.write(f"{k}: {v}")

    # ---------------------- Rule-Based Plan (always works) ----------------------
    st.markdown("### 📋 Core Plan")
    st.markdown(generate_plan(hormone_state, data))

    # ---------------------- Optional Gemini LLM ----------------------
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
            Condition: {hormone_state}
            Risk: {level_to_text(final)}

            Focus: {data['focus']}
            Foods: {", ".join(data['foods'])}

            Expand into:
            - Friendly explanation
            - Daily routine
            - 3-day Indian meal plan

            Keep safe, simple, non-medical.
            """

            response = model.generate_content(prompt)

            st.markdown("### 🤖 AI Enhanced Plan")
            st.markdown(response.text)

        except Exception as e:
            st.warning("AI plan unavailable. Showing core plan only.")

    else:
        st.info("Add GEMINI_API_KEY to enable AI enhancement.")

# ---------------------- Footer ----------------------
st.caption("This is general guidance and not a medical diagnosis.")
