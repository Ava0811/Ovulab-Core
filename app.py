import streamlit as st
import numpy as np
import io
from datetime import datetime
from openai import OpenAI

# PDF
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Ovulab",
    layout="wide"
)

# ---------------------------------------------------
# PROFESSIONAL UI
# ---------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background: linear-gradient(
            135deg,
            #fffdf8 0%,
            #f8f4ff 45%,
            #fff8fc 100%
        );
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #3f3f46;
    }

    /* Hero */

    .hero-title {
        font-size: 58px;
        font-weight: 700;
        color: #5b4b8a;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #7b7485;
        max-width: 760px;
        margin-bottom: 2.5rem;
        line-height: 1.6;
    }

    /* Glass Cards */

    .glass-card {
        background: rgba(255,255,255,0.72);
        backdrop-filter: blur(14px);

        border-radius: 26px;

        padding: 1.6rem;

        border: 1px solid rgba(255,255,255,0.35);

        box-shadow:
            0 8px 32px rgba(91,75,138,0.08);

        margin-bottom: 1.5rem;
    }

    /* Metric Cards */

    .metric-card {

        background: white;

        border-radius: 22px;

        padding: 1.2rem;

        text-align: center;

        border: 1px solid #f1e7ff;

        box-shadow:
            0 4px 18px rgba(0,0,0,0.04);
    }

    .metric-title {
        font-size: 14px;
        color: #8a7ca6;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #5b4b8a;
    }

    /* Buttons */

    .stButton > button {

        background: linear-gradient(
            135deg,
            #d8b4fe,
            #f9c5d5
        );

        color: #4b4453;

        border: none;

        border-radius: 16px;

        padding: 0.95rem 1.5rem;

        font-size: 15px;

        font-weight: 600;

        width: 100%;

        transition: all 0.3s ease;

        box-shadow:
            0 10px 25px rgba(216,180,254,0.25);

    }

    .stButton > button:hover {

        transform: translateY(-3px);

        box-shadow:
            0 14px 30px rgba(216,180,254,0.35);

    }

    /* Inputs */

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stSelectbox div[data-baseweb="select"] {

        border-radius: 16px !important;

        border: 1px solid #eadcff !important;

        background: rgba(255,255,255,0.75) !important;

        backdrop-filter: blur(10px);

        padding: 0.8rem !important;

        box-shadow:
            0 4px 16px rgba(0,0,0,0.03);

    }

    /* Chat */

    .user-bubble {

        background: #f5edff;

        padding: 1rem;

        border-radius: 18px;

        margin-bottom: 0.7rem;

        color: #4b4453;

    }

    .ai-bubble {

        background: #fff8fb;

        padding: 1rem;

        border-radius: 18px;

        margin-bottom: 1.2rem;

        border: 1px solid #f3ddeb;

        color: #4b4453;

    }

    /* Sidebar */

    section[data-testid="stSidebar"] {

        background: #faf7ff;

        border-right: 1px solid #efe8ff;

    }

    h1, h2, h3 {

        font-weight: 650;

        color: #51426d;

    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# API
# ---------------------------------------------------

OPENROUTER_API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ---------------------------------------------------
# SESSION
# ---------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.markdown("## Ovulab")

    st.markdown("""
    Personalized wellness support platform focused on:

    • Ovulation wellness  
    • PCOS-friendly nutrition  
    • Lifestyle support  
    • Personalized meal planning  
    • Biomineral recommendations  
    • Wellness tracking  
    """)

    st.markdown("---")

    st.info(
        "Ovulab provides educational wellness guidance and does not replace professional medical consultation."
    )

# ---------------------------------------------------
# HERO
# ---------------------------------------------------

st.markdown(
    """
    <div class='hero-title'>
        Ovulab
    </div>

    <div class='hero-subtitle'>
        Personalized wellness guidance designed to support healthier ovulation, nutrition and hormonal balance.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

st.markdown("## Wellness Profile")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Menstrual History")

    p1 = st.date_input(
        "Most recent period start"
    )

    p2 = st.date_input(
        "Previous period start"
    )

    p3 = st.date_input(
        "Earlier period start"
    )

    # Auto calculate cycle lengths

    l1 = abs((p1 - p2).days)

    l2 = abs((p2 - p3).days)

    l3 = l2

    st.caption(
        f"Detected cycle durations: {l1} days and {l2} days"
    )

    bleed = st.number_input(
        "Bleeding duration (days)",
        min_value=1.0,
        max_value=12.0,
        value=5.0
    )

with col2:

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=50,
        value=24
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30,
        max_value=150,
        value=60
    )

    height = st.number_input(
        "Height (cm)",
        min_value=120,
        max_value=220,
        value=160
    )

    activity = st.selectbox(
        "Activity Level",
        ["Low", "Moderate", "High"]
    )

diet = st.selectbox(
    "Diet Preference",
    ["Vegetarian", "Non-vegetarian"]
)

allergies = st.text_input(
    "Food allergies or intolerances",
    placeholder="Example: lactose intolerant, nut allergy"
)

calorie_goal = st.selectbox(
    "Goal",
    [
        "Weight Maintenance",
        "Weight Loss",
        "Weight Gain"
    ]
)

# ---------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------

height_m = height / 100

bmi = weight / (height_m ** 2)

if activity == "Low":
    activity_factor = 1.2

elif activity == "Moderate":
    activity_factor = 1.45

else:
    activity_factor = 1.7

bmr = 10 * weight + 6.25 * height - 5 * age - 161

maintenance_calories = int(bmr * activity_factor)

if calorie_goal == "Weight Loss":
    target_calories = maintenance_calories - 300

elif calorie_goal == "Weight Gain":
    target_calories = maintenance_calories + 300

else:
    target_calories = maintenance_calories

avg_cycle = np.mean([l1, l2, l3])

# ---------------------------------------------------
# METRIC CARDS
# ---------------------------------------------------

m1, m2, m3 = st.columns(3)

with m1:

    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>BMI</div>
            <div class='metric-value'>{bmi:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>Target Calories</div>
            <div class='metric-value'>{target_calories}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:

    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>Average Cycle</div>
            <div class='metric-value'>{avg_cycle:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# PDF
# ---------------------------------------------------

def generate_pdf(text):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    for line in text.split("\n"):

        if line.strip() == "":
            continue

        content.append(
            Paragraph(line, styles["Normal"])
        )

        content.append(
            Spacer(1, 10)
        )

    doc.build(content)

    buffer.seek(0)

    return buffer

# ---------------------------------------------------
# AI FUNCTION
# ---------------------------------------------------

def generate_ai_report(prompt):

    try:

        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return completion.choices[0].message.content

    except Exception as e:

        return f"""
AI API Error

{e}
"""

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------

left_col, right_col = st.columns([1.35, 0.85])

# ---------------------------------------------------
# REPORT SECTION
# ---------------------------------------------------

with left_col:

    st.markdown("## Personalized Wellness Report")

    if st.button("Generate Wellness Plan"):

        avg = np.mean([l1, l2, l3])

        std = np.std([l1, l2, l3])

        if 26 <= avg <= 32 and std < 3:
            risk = "Low"
            pattern = "Stable ovulatory pattern"

        elif std > 7 or avg < 22 or avg > 35:
            risk = "High"
            pattern = "Hormonal variability"

        else:
            risk = "Medium"
            pattern = "Mild ovulatory irregularity"

        prompt = f"""
You are Ovulab AI.

Generate a professional wellness report.

User Data:
- Cycle lengths: {l1}, {l2}, {l3}
- Average cycle: {avg:.1f}
- Variation: {std:.2f}
- Bleeding duration: {bleed}
- Age: {age}
- Weight: {weight}
- Height: {height}
- BMI: {bmi:.1f}
- Activity level: {activity}
- Diet: {diet}
- Allergies: {allergies}
- Calories: {target_calories}
- Risk: {risk}
- Pattern: {pattern}

Generate:
- ovulation wellness analysis
- physiological insight
- vitamins and biominerals
- personalized nutrition strategy
- Indian meal plan
- calorie-aware meals
- hydration guidance
- exercise and lifestyle recommendations
- supportive wellness guidance

Avoid diagnosing disease.
Keep tone professional, calming and educational.
"""

        with st.spinner("Generating personalized wellness report..."):

            result = generate_ai_report(prompt)

        st.markdown(
            f"""
            <div class='glass-card'>
            {result}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.session_state.history.append({
            "time": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "result": result
        })

        pdf = generate_pdf(result)

        st.download_button(
            label="Download Wellness Report",
            data=pdf,
            file_name="ovulab_report.pdf",
            mime="application/pdf"
        )

# ---------------------------------------------------
# CHATBOT
# ---------------------------------------------------

with right_col:

    st.markdown("## Wellness Assistant")

    user_query = st.text_input(
        "Ask about nutrition, vitamins, ovulation wellness or lifestyle support"
    )

    if st.button("Submit Question"):

        chatbot_prompt = f"""
You are Ovulab AI.

User Question:
{user_query}

Provide supportive wellness guidance focused on:
- ovulation wellness
- PCOS-friendly nutrition
- vitamins and biominerals
- calorie-aware eating
- realistic Indian lifestyle habits
- stress management
- exercise support

Avoid medical diagnosis.
Keep tone calm, professional and supportive.
"""

        response = generate_ai_report(chatbot_prompt)

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": response
        })

    for chat in st.session_state.chat_history[::-1]:

        st.markdown(
            f"""
            <div class='user-bubble'>
            <strong>User</strong><br><br>
            {chat["question"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class='ai-bubble'>
            <strong>Ovulab</strong><br><br>
            {chat["answer"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------

st.markdown("## Previous Wellness Reports")

for i, record in enumerate(st.session_state.history[::-1]):

    with st.expander(
        f"Report {i+1} ({record['time']})"
    ):

        st.markdown(record["result"])

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.caption(
    "Ovulab provides educational wellness guidance and does not replace professional medical consultation."
)
