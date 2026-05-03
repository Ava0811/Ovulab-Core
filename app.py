# app.py — Ovulab (LLM Weekly Plan + PDF Export)

import streamlit as st
import requests
import numpy as np
import io
from datetime import datetime

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------- PAGE ----------------------
st.set_page_config(page_title="Ovulab LLM", page_icon="💜", layout="centered")

# ---------------------- MEMORY ----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------- UI ----------------------
st.title("💜 Ovulab — LLM Weekly Planner")

st.caption("Cycle length = days between start of one period and next")

# ---------------------- INPUT ----------------------
l1 = st.number_input("Last cycle length", 10, 90, 28)
l2 = st.number_input("Previous cycle length", 10, 90, 29)
l3 = st.number_input("Earlier cycle length", 10, 90, 27)

bleed = st.number_input("Bleeding duration (days)", 1.0, 12.0, 5.0)

diet = st.selectbox("Diet preference", ["Vegetarian", "Non-vegetarian"])

# ---------------------- LLM FUNCTION ----------------------
def ask_llm(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        return res.json()["response"]
    except:
        return "⚠️ LLM not running. Run: ollama run mistral"

# ---------------------- PDF FUNCTION ----------------------
def generate_pdf(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    for line in text.split("\n"):
        if line.strip() == "":
            continue
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ---------------------- BUTTON ----------------------
if st.button("Analyze"):

    avg = np.mean([l1, l2, l3])
    std = np.std([l1, l2, l3])

    prompt = f"""
You are Ovulab, an AI menstrual health assistant trained in physiology and nutritional biochemistry.

User Data:
Cycle lengths: {l1}, {l2}, {l3}
Average: {avg:.1f}
Variation: {std:.2f}
Bleeding days: {bleed}
Diet: {diet}

Tasks:

1. Classify risk (LOW / MEDIUM / HIGH)
2. Infer physiological pattern
3. Generate structured output:

---

### 🧬 Cycle Analysis
...

### 🧠 Insight
...

### 🥗 Nutrition Strategy
...

### 🍽️ Food Recommendations
...

### 🧘 Lifestyle Plan
...

### 🗓️ Weekly Meal Plan (7 Days)

Each day MUST include:
Breakfast:
Lunch:
Dinner:
Snack:

Rules:
- Each day must be different
- Avoid repetition
- Use Indian foods
- Respect diet preference

### 🔁 Monthly Guidance
Explain how to rotate this weekly plan over 4 weeks

---

Keep tone simple and safe.
End with disclaimer.
"""

    result = ask_llm(prompt)

    # Display
    st.markdown("### 🤖 AI Weekly Plan")
    st.markdown(result)

    # Save to memory
    st.session_state.history.append({
        "time": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "cycles": [l1, l2, l3],
        "bleeding": bleed,
        "result": result
    })

    # PDF export
    pdf = generate_pdf(result)

    st.download_button(
        "📄 Download Weekly Plan PDF",
        pdf,
        "ovulab_weekly_plan.pdf",
        "application/pdf"
    )

# ---------------------- HISTORY ----------------------
st.markdown("### 🧠 Past Records")

for i, record in enumerate(st.session_state.history[::-1]):
    with st.expander(f"Record {i+1} ({record['time']})"):
        st.write("Cycles:", record["cycles"])
        st.write("Bleeding:", record["bleeding"])
        st.write(record["result"])

# ---------------------- FOOTER ----------------------
st.caption("This is general guidance and not a medical diagnosis.")
