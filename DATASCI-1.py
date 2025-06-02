import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("student_mental_health.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

data = load_data()
st.title("🎓 Student Mental Health Dashboard & Predictor")

# Tabs: Dashboard | Prediction
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Prediction"])

with tab1:
    st.header("Complete Dataset")
    st.dataframe(data)

    # Plot 1: Gender distribution
    if "gender" in data.columns:
        st.subheader("Distribution by Gender")
        st.bar_chart(data["gender"].value_counts())

    # Plot 2: Treatment by Gender
    if "gender" in data.columns and "treatment" in data.columns:
        st.subheader("Treatment Need by Gender")
        treatment_by_gender = data.groupby("gender")["treatment"].value_counts().unstack().fillna(0)
        st.bar_chart(treatment_by_gender)

    # Plot 3: Correlation Heatmap
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

    required_cols = {"stress_level", "screen_time", "academic_performance_change", "treatment"}
    if required_cols.issubset(data.columns):
        df = data.copy()

        # Encode categorical features if needed
        for col in ["stress_level", "academic_performance_change", "treatment"]:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category").cat.codes

        # Features and target
        X = df[["stress_level", "screen_time", "academic_performance_change"]]
        y = df["treatment"]

        # Model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Sidebar inputs
        st.sidebar.header("Student Lifestyle Input")
        stress = st.sidebar.slider("Stress Level (1 = Low, 5 = High)", 1, 5, 3)
        screen = st.sidebar.slider("Daily Screen Time (hours)", 0.0, 16.0, 6.0, 0.5)
        performance = st.sidebar.selectbox("Academic Performance Change", ["Improved", "No Change", "Declined"])

        # Encode input
        perf_map = {"Improved": 0, "No Change": 1, "Declined": 2}
        input_df = pd.DataFrame([{
            "stress_level": stress,
            "screen_time": screen,
            "academic_performance_change": perf_map[performance]
        }])

        if st.sidebar.button("Predict"):
            prediction = model.predict(input_df)[0]
            result = "Needs Treatment" if prediction == 1 else "Does Not Need Treatment"
            st.success(f"🧠 Based on the input, the model predicts: **{result}**")
    else:
        st.error("Required columns 'stress_level', 'screen_time', 'academic_performance_change', or 'treatment' are missing.")
