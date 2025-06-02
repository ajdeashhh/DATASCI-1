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

    required_cols = {'stress_level', 'academic_performance_change', 'treatment'}

    # Show columns to debug
    st.write("Columns in dataset:", data.columns.tolist())

    if required_cols.issubset(data.columns):
        df = data.copy()

        # Encode categorical columns if needed
        for col in ['stress_level', 'academic_performance_change', 'treatment']:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes

        X = df[['stress_level', 'academic_performance_change']]
        y = df['treatment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        st.sidebar.header("Student Input")
        stress_level = st.sidebar.slider("Stress Level (1=Low, 5=High)", 1, 5, 3)
        perf_options = ["Improved", "No Change", "Declined"]
        academic_performance_change = st.sidebar.selectbox("Academic Performance Change", perf_options)

        perf_map = {name: code for code, name in enumerate(perf_options)}

        input_df = pd.DataFrame([{
            'stress_level': stress_level,
            'academic_performance_change': perf_map[academic_performance_change]
        }])

        if st.sidebar.button("Predict"):
            prediction = model.predict(input_df)[0]
            result = "Needs Treatment" if prediction == 1 else "Does Not Need Treatment"
            st.success(f"Based on the input, the model predicts: **{result}**")

    else:
        missing = required_cols.difference(data.columns)
        st.error(f"Missing columns in dataset: {', '.join(missing)}")

