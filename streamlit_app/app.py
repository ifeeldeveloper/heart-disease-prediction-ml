import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration

st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide"
)

# Load Dataset


@st.cache_data
def load_data():
    return pd.read_excel("data/heart_disease_cleaned.xlsx")


heart_df = load_data()

# Load Machine Learning Objects


@st.cache_resource
def load_model():
    return (
        joblib.load("models/heart_stroke_model.pkl"),
        joblib.load("models/scaler.pkl"),
        joblib.load("models/gender_encoder.pkl"),
        joblib.load("models/education_encoder.pkl"),
        joblib.load("models/exercise_encoder.pkl"),
        joblib.load("models/agegroup_encoder.pkl"),
    )


model, scaler, gender_encoder, education_encoder, exercise_encoder, agegroup_encoder = load_model()

# Title

st.title("❤️ Heart Disease Dashboard")
st.write("Interactive dashboard for exploring the Heart Disease dataset.")

# Navigation

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🤖 Prediction"]
)

# Dashboard Page

if page == "📊 Dashboard":

    # Sidebar Filters

    st.sidebar.header("Filters")

    gender = st.sidebar.selectbox(
        "Gender",
        ["All"] + sorted(heart_df["Gender"].unique().tolist())
    )

    age_group = st.sidebar.selectbox(
        "Age Group",
        ["All"] + sorted(heart_df["AgeGroup"].astype(str).unique().tolist())
    )

    smoker = st.sidebar.selectbox(
        "Current Smoker",
        ["All", 0, 1]
    )

    diabetes = st.sidebar.selectbox(
        "Diabetes",
        ["All", 0, 1]
    )

    # Apply Filters

    filtered_df = heart_df.copy()

    if gender != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == gender]

    if age_group != "All":
        filtered_df = filtered_df[filtered_df["AgeGroup"].astype(
            str) == age_group]

    if smoker != "All":
        filtered_df = filtered_df[filtered_df["currentSmoker"] == smoker]

    if diabetes != "All":
        filtered_df = filtered_df[filtered_df["Diabetes"] == diabetes]

    # KPI Metrics

    st.subheader("📊 Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Participants",
        len(filtered_df)
    )

    col2.metric(
        "Average Age",
        round(filtered_df["Age"].mean(), 1)
    )

    col3.metric(
        "Average BMI",
        round(filtered_df["BMI"].mean(), 1)
    )

    col4.metric(
        "Average Cholesterol",
        round(filtered_df["totChol"].mean(), 1)
    )

    # Charts

    left, right = st.columns(2)

    # Gender Distribution

    with left:

        st.subheader("Gender Distribution")

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.countplot(
            data=filtered_df,
            x="Gender",
            ax=ax
        )

        plt.tight_layout()

        st.pyplot(fig)

    # Age Group Distribution

    with right:

        st.subheader("Age Group Distribution")

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.countplot(
            data=filtered_df,
            x="AgeGroup",
            order=sorted(filtered_df["AgeGroup"].astype(str).unique()),
            ax=ax
        )

        plt.xticks(rotation=30)

        plt.tight_layout()

        st.pyplot(fig)

    left, right = st.columns(2)

    # BMI Histogram

    with left:

        st.subheader("BMI Distribution")

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.histplot(
            filtered_df["BMI"],
            bins=15,
            kde=True,
            ax=ax
        )

        plt.tight_layout()

        st.pyplot(fig)

    # Cholesterol Histogram

    with right:

        st.subheader("Cholesterol Distribution")

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.histplot(
            filtered_df["totChol"],
            bins=15,
            kde=True,
            ax=ax
        )

        plt.tight_layout()

        st.pyplot(fig)

    # Heart Disease Distribution

    st.subheader("Heart Disease Distribution")

    heart_counts = filtered_df["heartStroke"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        heart_counts,
        labels=["No", "Yes"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Heart Disease")

    st.pyplot(fig)

    # Dataset

    st.subheader("Filtered Dataset")

    st.dataframe(filtered_df)


elif page == "🤖 Prediction":

    st.header("🤖 Heart Stroke Prediction")

    st.success("✅ Machine Learning model loaded successfully!")

    # User Input Form

    st.subheader("Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        age = st.number_input(
            "Age",
            min_value=20,
            max_value=100,
            value=45
        )

        education_display = st.selectbox(
            "Education",
            [
                "Uneducated",
                "Primary School",
                "Graduate",
                "Postgraduate"
            ]
        )

        education_mapping = {
            "Uneducated": "uneducated",
            "Primary School": "primaryschool",
            "Graduate": "graduate",
            "Postgraduate": "postgraduate"
        }

        education = education_mapping[education_display]

        current_smoker = st.selectbox(
            "Current Smoker",
            [0, 1]
        )

        cigs_per_day = st.number_input(
            "Cigarettes Per Day",
            min_value=0,
            max_value=80,
            value=0
        )

        bp_meds = st.selectbox(
            "BP Medication",
            [0, 1]
        )

        diabetes = st.selectbox(
            "Diabetes",
            [0, 1]
        )

    with col2:

        prevalent_stroke = st.selectbox(
            "Previous Stroke",
            [0, 1]
        )

        prevalent_hyp = st.selectbox(
            "Hypertension",
            [0, 1]
        )

        total_chol = st.number_input(
            "Total Cholesterol",
            min_value=100,
            max_value=700,
            value=200
        )

        sys_bp = st.number_input(
            "Systolic BP",
            min_value=80,
            max_value=250,
            value=120
        )

        dia_bp = st.number_input(
            "Diastolic BP",
            min_value=40,
            max_value=150,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0
        )

        heart_rate = st.number_input(
            "Heart Rate",
            min_value=40,
            max_value=180,
            value=75
        )

        glucose = st.number_input(
            "Glucose",
            min_value=40,
            max_value=500,
            value=90
        )

        exercise = st.selectbox(
            "Exercise Frequency",
            [
                "Daily",
                "Weekly",
                "Monthly"
            ]
        )

    # Encode Categorical Variables

    gender = gender_encoder.transform([gender])[0]

    education = education_encoder.transform([education])[0]

    exercise = exercise_encoder.transform([exercise])[0]

    # Create AgeGroup Automatically

    if 30 <= age < 40:
        age_group = "30-39"

    elif 40 <= age < 50:
        age_group = "40-49"

    elif 50 <= age < 60:
        age_group = "50-59"

    else:
        age_group = "60-70"

    age_group = agegroup_encoder.transform([age_group])[0]

    # Create Input DataFrame

    input_data = pd.DataFrame({

        "Gender": [gender],
        "Age": [age],
        "Education": [education],
        "currentSmoker": [current_smoker],
        "cigsPerDay": [cigs_per_day],
        "BPMeds": [bp_meds],
        "prevalentStroke": [prevalent_stroke],
        "prevalentHyp": [prevalent_hyp],
        "Diabetes": [diabetes],
        "totChol": [total_chol],
        "sysBP": [sys_bp],
        "diaBP": [dia_bp],
        "BMI": [bmi],
        "heartRate": [heart_rate],
        "Glucose": [glucose],
        "Exercise": [exercise],
        "AgeGroup": [age_group]

    })

    st.divider()

    st.subheader("Prepared Input")

    st.dataframe(input_data)

    # Scale the Input

    input_scaled = scaler.transform(input_data)

    # Predict Button

    if st.button("🔍 Predict Heart Disease Risk"):

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")

        else:
            st.success("✅ Low Risk of Heart Disease")

        st.metric(
            "Prediction Probability",
            f"{probability:.2%}"
        )

        st.caption(
            "⚠️ Disclaimer: This application is developed for educational purposes only. "
            "Predictions generated by this model are not medical diagnoses and should not "
            "replace professional healthcare advice."
        )
