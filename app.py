import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import os

st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #f7f8fc; }

    .card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 24px 28px;
        border: 1px solid #e8eaf0;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .score-display {
        font-size: 72px;
        font-weight: 600;
        line-height: 1;
        color: #1a1a2e;
        font-family: 'DM Mono', monospace;
    }
    .grade-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 50px;
        font-size: 15px;
        font-weight: 500;
        margin-top: 10px;
    }
    [data-testid="stMetricLabel"] { font-size: 13px; color: #6b7280; font-weight: 500; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 600; color: #1a1a2e; }
    .page-title { font-size: 32px; font-weight: 600; color: #1a1a2e; margin: 0; }
    .page-subtitle { font-size: 15px; color: #6b7280; margin-top: 4px; }
    .section-heading {
        font-size: 16px; font-weight: 600; color: #1a1a2e;
        margin-bottom: 16px; padding-bottom: 8px;
        border-bottom: 2px solid #f0f0f5;
    }
    
    /* Slider label styling */
    [data-testid="stSlider"] label { color: #1a1a2e !important; font-weight: 600 !important; font-size: 14px !important; }
    [data-testid="stSlider"] span { color: #1a1a2e !important; }
    [role="slider"] { color: #1a1a2e !important; }
    
    /* General text visibility */
    label { color: #1a1a2e !important; }
    .stSlider p { color: #1a1a2e !important; }
    .stSlider span { color: #1a1a2e !important; }
    [data-testid*="slider"] { color: #1a1a2e !important; }
    
    .insight-item {
        background: #f7f8fc;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13.5px;
        color: #374151;
        border-left: 3px solid;
        margin-bottom: 8px;
    }
    .divider { border: none; border-top: 1px solid #e8eaf0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "student_final_score_predictor", "Students_Performance_Synthetic_5000.csv")
    data = pd.read_csv(csv_path)

    FEATURES = [
        "Study_Hours_per_Week", "Attendance (%)",
        "Midterm_Score", "Assignments_Avg", "Quizzes_Avg",
    ]
    TARGET = "Final_Score"

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    poly = PolynomialFeatures(degree=2, include_bias=True)
    scaler = StandardScaler()

    X_train_poly   = poly.fit_transform(X_train)
    X_test_poly    = poly.transform(X_test)      
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled  = scaler.transform(X_test_poly) 
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "MSE": mean_squared_error(y_test, y_pred_test),
        "R2":  r2_score(y_test, y_pred_test),
    }

    return model, poly, scaler, data, FEATURES, metrics


model, poly, scaler, data, FEATURES, metrics = load_and_train_model()


def get_grade(score: float):
    """Returns (grade_label, background_color, text_color)."""
    if score >= 85:
        return "Excellent",        "#d1fae5", "#065f46"
    elif score >= 75:
        return "Good",             "#dbeafe", "#1e40af"
    elif score >= 65:
        return "Average",          "#fef9c3", "#854d0e"
    elif score >= 55:
        return "Below Average",    "#ffedd5", "#9a3412"
    else:
        return "Needs Improvement","#f7f3f3", "#991b1b"


def predict_score(study_hours, attendance, midterm, assignments, quizzes):
    raw    = np.array([[study_hours, attendance, midterm, assignments, quizzes]])
    poly_f = poly.transform(raw)
    scaled = scaler.transform(poly_f)
    score  = model.predict(scaled)[0]
    return float(np.clip(score, 0.0, 100.0))  
with st.sidebar:
    st.markdown("### 🎓 About This App")
    st.markdown(
        "Uses a **Polynomial Regression** model (degree 2) "
        "trained on 5,000 synthetic student records."
    )
    st.markdown("---")
    st.markdown("### 📐 Model Performance")
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| R² Score | `{metrics['R2']:.4f}` |
| MAE | `{metrics['MAE']:.2f} pts` |
| MSE | `{metrics['MSE']:.2f}` |
""")
    st.markdown("---")
    st.markdown("### 📂 Dataset Summary")
    st.markdown(f"""
- **Students:** {len(data):,}
- **Features:** {len(FEATURES)}
- **Train / Test:** 80% / 20%
- **Avg final score:** {data['Final_Score'].mean():.1f}
""")
    st.caption("Built with Streamlit · Scikit-learn · Matplotlib")


st.markdown("""
<p class="page-title">🎓 Student Score Predictor</p>
<p class="page-subtitle">Adjust the sliders to see how each factor influences the predicted final exam score.</p>
""", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


left_col, right_col = st.columns([1.1, 1.0], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">📋 Student Input Parameters</p>', unsafe_allow_html=True)

    study_hours = st.slider(
        "📚 Study Hours per Week",
        min_value=5.0, max_value=30.0, value=17.0, step=0.5,
        help="Hours dedicated to studying outside class each week."
    )
    attendance = st.slider(
        "🗓️ Attendance (%)",
        min_value=50.0, max_value=100.0, value=74.0, step=0.5,
        help="Percentage of classes attended during the semester."
    )
    midterm = st.slider(
        "📝 Midterm Score",
        min_value=40.0, max_value=100.0, value=75.0, step=1.0,
        help="Score obtained in the midterm examination (out of 100)."
    )
    assignments = st.slider(
        "📋 Assignments Average",
        min_value=50.0, max_value=100.0, value=75.0, step=1.0,
        help="Average score across all submitted assignments."
    )
    quizzes = st.slider(
        "🧪 Quizzes Average",
        min_value=50.0, max_value=100.0, value=79.0, step=1.0,
        help="Average score across all in-class quizzes."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    prediction = predict_score(study_hours, attendance, midterm, assignments, quizzes)
    grade_label, bg_color, text_color = get_grade(prediction)
    avg = data["Final_Score"].mean()
    diff = prediction - avg
    diff_str  = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
    diff_color = "#065f46" if diff >= 0 else "#991b1b"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">🎯 Predicted Final Score</p>', unsafe_allow_html=True)
    st.markdown(f"""
<div style="text-align:center; padding:12px 0 8px 0;">
    <div class="score-display">{prediction:.1f}</div>
    <div style="font-size:13px; color:#9ca3af; margin-top:4px;">out of 100</div>
    <div class="grade-badge" style="background:{bg_color}; color:{text_color};">{grade_label}</div>
</div>
""", unsafe_allow_html=True)
    st.progress(prediction / 100)
    st.markdown(f"""
<p style="text-align:center; font-size:13px; color:#6b7280; margin-top:8px;">
    <span style="color:{diff_color}; font-weight:600;">{diff_str} pts</span>
    vs. dataset average ({avg:.1f})
</p>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">💡 Smart Insights</p>', unsafe_allow_html=True)

    insights = []
    if study_hours < 12:
        insights.append(("📚", "Increasing study hours above 12/week typically raises scores.", "#6366f1"))
    if attendance < 70:
        insights.append(("🗓️", "Attendance below 70% strongly correlates with lower finals.", "#ef4444"))
    if midterm < 60:
        insights.append(("📝", "A midterm below 60 is a strong risk signal for the final.", "#f59e0b"))
    if assignments < 65:
        insights.append(("📋", "Completing assignments consistently boosts predicted scores.", "#10b981"))
    if quizzes < 65:
        insights.append(("🧪", "Quiz performance reflects retention — aim above 70.", "#3b82f6"))
    if not insights:
        insights.append(("✅", "All factors look healthy. This student is on track!", "#10b981"))

    for icon, text, color in insights:
        st.markdown(
            f'<div class="insight-item" style="border-left-color:{color};">{icon} {text}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('<p class="section-heading">📊 Dataset Overview</p>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Students",    f"{len(data):,}")
m2.metric("Average Final Score", f"{data['Final_Score'].mean():.1f}")
m3.metric("Highest Score",     f"{data['Final_Score'].max():.1f}")
m4.metric("Model R² Score",    f"{metrics['R2']:.4f}")

chart_col1, chart_col2 = st.columns(2, gap="medium")

with chart_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">📈 Score Distribution</p>', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    fig1.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#f7f8fc")

    ax1.hist(data["Final_Score"], bins=30, color="#6366f1",
             edgecolor="white", linewidth=0.6, alpha=0.85)
    ax1.axvline(prediction, color="#ef4444", linestyle="--",
                linewidth=2, label=f"Your student: {prediction:.1f}")
    ax1.axvline(avg, color="#10b981", linestyle="--",
                linewidth=2, label=f"Dataset avg: {avg:.1f}")

    ax1.set_xlabel("Final Score", fontsize=11, color="#374151")
    ax1.set_ylabel("Number of Students", fontsize=11, color="#374151")
    ax1.tick_params(colors="#6b7280", labelsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color("#e5e7eb")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, color="#e5e7eb")
    plt.tight_layout()
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">🔍 Your Input vs. Dataset Average</p>', unsafe_allow_html=True)

    student_vals = [study_hours, attendance, midterm, assignments, quizzes]
    dataset_avgs = [
        data["Study_Hours_per_Week"].mean(),
        data["Attendance (%)"].mean(),
        data["Midterm_Score"].mean(),
        data["Assignments_Avg"].mean(),
        data["Quizzes_Avg"].mean(),
    ]
    maxes        = [30, 100, 100, 100, 100]
    norm_student = [v / m * 100 for v, m in zip(student_vals, maxes)]
    norm_avg     = [v / m * 100 for v, m in zip(dataset_avgs, maxes)]

    short_labels = ["Study\nHours", "Attend\n(%)", "Midterm", "Assign.", "Quizzes"]
    x = np.arange(len(short_labels))
    width = 0.35

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#f7f8fc")

    ax2.bar(x - width/2, norm_student, width, color="#6366f1", label="This student", alpha=0.85)
    ax2.bar(x + width/2, norm_avg,     width, color="#d1d5db", label="Dataset avg",  alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=9, color="#374151")
    ax2.set_ylabel("Normalised value (%)", fontsize=11, color="#374151")
    ax2.tick_params(colors="#6b7280", labelsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color("#e5e7eb")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, color="#e5e7eb")
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.caption(
    f"🎓 Student Score Predictor · Polynomial Regression (degree=2) · "
    f"Trained on {len(data):,} records · R² = {metrics['R2']:.4f}"
)
