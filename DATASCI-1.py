import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")

# Load and display full dataset
@st.cache_data
def load_data():
    df = pd.read_csv("student_mental_health.csv")
    return df

data = load_data()
st.title("🎓 Student Mental Health Dashboard & Predictor")

# Tabs: Dashboard | Prediction
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Prediction"])

with tab1:
    st.header("Complete Dataset")
    st.dataframe(data)

    # Plot 1: Gender distribution
    if "Gender" in data.columns:
        st.subheader("Distribution by Gender")
        gender_counts = data["Gender"].value_counts()
        st.bar_chart(gender_counts)

    # Plot 2: Treatment by Gender (if both columns exist)
    if "Gender" in data.columns and "treatment" in data.columns:
        st.subheader("Treatment Need by Gender")
        treatment_by_gender = data.groupby("Gender")["treatment"].value_counts().unstack().fillna(0)
        st.bar_chart(treatment_by_gender)

    # Plot 3: Heatmap
    st.subheader("Correlation Heatmap")
    df_numeric = data.select_dtypes(include='number')
    if not df_numeric.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns to compute correlation.")

with tab2:
    st.header("Predict Mental Health Treatment Need")

    st.markdown("### Enter your details below:")

    screen_time = st.slider(
        "Screen Time (hours per day)",
        0.0, 16.0, 6.0, 0.5,
        help="Average daily screen time in hours."
    )

    sleep_duration = st.slider(
        "Sleep Duration (hours)",
        0.0, 12.0, 7.0, 0.5,
        help="Average daily sleep duration in hours."
    )

    physical_activity = st.slider(
        "Physical Activity (hours per week)",
        0.0, 20.0, 3.0, 0.5,
        help="Total hours of physical activity per week."
    )

    stress_level = st.slider(
        "Stress Level (1 = Low, 5 = High)",
        1, 5, 3,
        help="Rate your current stress level on a scale from 1 (lowest) to 5 (highest)."
    )

    anxious_before_exams = st.radio(
        "Do you feel anxious before exams?",
        options=["Yes", "No"],
        index=1,
        help="Select if you often feel anxious before exams."
    )

    perf_options = {
        "Improved": "Your academic performance has improved recently.",
        "No Change": "Your academic performance has stayed about the same.",
        "Declined": "Your academic performance has declined recently."
    }
    academic_performance_change = st.selectbox(
        "Academic Performance Change",
        options=list(perf_options.keys()),
        help="Select how your academic performance has changed recently."
    )
    st.caption(perf_options[academic_performance_change])

    if st.button("Predict"):
        # Rule-based logic
        needs_treatment = False

        if stress_level >= 4:
            needs_treatment = True
        if anxious_before_exams == "Yes":
            needs_treatment = True
        if academic_performance_change == "Declined":
            needs_treatment = True
        if physical_activity < 2:
            needs_treatment = True
        if sleep_duration < 6:
            needs_treatment = True

        result = "Needs Treatment" if needs_treatment else "Does Not Need Treatment"
        st.success(f"Based on your inputs, the prediction is: **{result}**")
