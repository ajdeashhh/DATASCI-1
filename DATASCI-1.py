import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("student_mental_health.csv")
    # Clean column names for consistency
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

data = load_data()
st.title("🎓 Student Mental Health Dashboard & Predictor")

# Check column names
st.write("🧾 Column names detected in dataset:", data.columns.tolist())

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Prediction"])

with tab1:
    st.header("Complete Dataset")
    st.dataframe(data)

    # Plot 1: Gender distribution
    if "gender" in data.columns:
        st.subheader("Distribution by Gender")
        gender_counts = data["gender"].value_counts()
        st.bar_chart(gender_counts)

    # Plot 2: Treatment by Gender
    if "gender" in data.columns and "treatment" in data.columns:
        st.subheader("Treatment Need by Gender")
        treatment_by_gender = data.groupby("gender")["treatment"].value_counts().unstack().fillna(0)
        st.bar_chart(treatment_by_gender)

    # Plot 3: Correlation heatmap
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

    # Check necessary columns exist
    if {"screen_time", "sleep_duration", "treatment"}.issubset(data.columns):
        X = data[["screen_time", "sleep_duration"]]
        y = data["treatment"]

        # Encode target if it's categorical
        if y.dtype == "object":
            y = y.astype("category").cat.codes

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Sidebar input
        st.sidebar.header("Student Lifestyle Input")
        screen_time = st.sidebar.slider("Average Daily Screen Time (hrs)", 0.0, 16.0, 6.0, 0.5)
        sleep_duration = st.sidebar.slider("Average Sleep Duration (hrs)", 0.0, 12.0, 7.0, 0.5)

        if st.sidebar.button("Predict"):
            input_df = pd.DataFrame([[screen_time, sleep_duration]], columns=["screen_time", "sleep_duration"])
            prediction = model.predict(input_df)[0]
            result = "Needs Treatment" if prediction == 1 else "Does Not Need Treatment"
            st.success(f"🧠 The model predicts: **{result}**")
    else:
        st.error("❌ Required columns 'screen_time', 'sleep_duration', or 'treatment' are missing from your dataset.")
