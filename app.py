import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Job Acceptance Prediction", page_icon="🎯", layout="wide")

@st.cache_resource
def get_engine():
    pw = quote_plus("WelcomeGlobal1@")
    return create_engine(f"mysql+mysqlconnector://root:{pw}@127.0.0.1:3306/job_prediction_db")

@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM candidates", con=engine)
    return df

@st.cache_resource
def load_model():
    with open("data/best_model.pkl", "rb") as f:
        return pickle.load(f)

def show_kpis(df):
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    total = len(df)
    placed = (df["status"].str.lower() == "placed").sum()
    placement_rate = placed/total*100
    acceptance_rate = df["target"].mean()*100
    avg_interview = df["Interview_Score_Avg"].mean()
    avg_skills = df["skills_match_percentage"].mean()
    dropout = 100 - acceptance_rate
    high_risk = (df["Interview_Score_Avg"] < 50).sum()/total*100

    col1.metric("👥 Total Candidates", f"{total:,}")
    col2.metric("✅ Placement Rate", f"{placement_rate:.1f}%")
    col3.metric("🤝 Acceptance Rate", f"{acceptance_rate:.1f}%")
    col4.metric("📊 Avg Interview Score", f"{avg_interview:.1f}")
    col5.metric("🎯 Avg Skills Match", f"{avg_skills:.1f}%")
    col6.metric("⚠️ High Risk %", f"{high_risk:.1f}%")

def tab_eda(df):
    st.subheader("📊 Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        status = df["status"].value_counts().reset_index()
        status.columns = ["Status","Count"]
        fig = px.pie(status, names="Status", values="Count",
                     title="Placement Status Distribution",
                     color_discrete_sequence=["#2ecc71","#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="Interview_Score_Avg", color="status",
                           title="Interview Score Distribution by Status",
                           barmode="overlay", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        comp = df.groupby("company_tier")["target"].mean().reset_index()
        comp.columns = ["Company Tier","Acceptance Rate"]
        comp["Acceptance Rate"] = comp["Acceptance Rate"]*100
        fig = px.bar(comp, x="Company Tier", y="Acceptance Rate",
                     title="Acceptance Rate by Company Tier",
                     color="Acceptance Rate", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(df, x="Experience_Category", y="skills_match_percentage",
                     color="status", title="Skills Match by Experience Category")
        st.plotly_chart(fig, use_container_width=True)
    c5, c6 = st.columns(2)
    with c5:
        exp = df.groupby("Experience_Category")["target"].mean().reset_index()
        exp["target"] = exp["target"]*100
        fig = px.bar(exp, x="Experience_Category", y="target",
                     title="Placement Rate by Experience",
                     color="target", color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        comp_lvl = df.groupby("competition_level")["target"].mean().reset_index()
        comp_lvl["target"] = comp_lvl["target"]*100
        fig = px.bar(comp_lvl, x="competition_level", y="target",
                     title="Acceptance Rate by Competition Level",
                     color="target", color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)

def tab_ml(df):
    st.subheader("🤖 ML Model Performance")
    results = pd.read_csv("data/model_results.csv")
    feature_imp = pd.read_csv("data/feature_importance.csv")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(results, x="Model", y="Accuracy",
                     title="Model Accuracy Comparison",
                     color="Accuracy", color_continuous_scale="Blues",
                     text=results["Accuracy"].apply(lambda x: f"{x*100:.2f}%"))
        fig.update_layout(yaxis_range=[0.8, 0.95])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(feature_imp, x="Importance", y="Feature",
                     orientation="h", title="Top 10 Feature Importance",
                     color="Importance", color_continuous_scale="Reds")
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Model Metrics")
    best = results.loc[results["Accuracy"].idxmax()]
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Model", best["Model"])
    col2.metric("✅ Best Accuracy", f"{best['Accuracy']*100:.2f}%")
    col3.metric("📊 Total Models", len(results))

def tab_predict(df):
    st.subheader("🔮 Predict Job Acceptance")
    st.markdown("Enter candidate details to predict job acceptance probability:")

    model_data = load_model()
    model = model_data["model"]

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 60, 25)
        ssc = st.slider("SSC %", 40.0, 100.0, 70.0)
        hsc = st.slider("HSC %", 40.0, 100.0, 70.0)
        degree = st.slider("Degree %", 40.0, 100.0, 70.0)
        technical = st.slider("Technical Score", 0.0, 100.0, 60.0)
        aptitude = st.slider("Aptitude Score", 0.0, 100.0, 60.0)
        communication = st.slider("Communication Score", 0.0, 100.0, 60.0)
        skills = st.slider("Skills Match %", 0.0, 100.0, 60.0)

    with c2:
        certs = st.number_input("Certifications Count", 0, 20, 2)
        exp = st.number_input("Years of Experience", 0, 30, 2)
        prev_ctc = st.number_input("Previous CTC (LPA)", 0.0, 50.0, 5.0)
        exp_ctc = st.number_input("Expected CTC (LPA)", 0.0, 100.0, 10.0)
        notice = st.number_input("Notice Period (Days)", 0, 180, 30)
        emp_gap = st.number_input("Employment Gap (Months)", 0, 60, 0)

    with c3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        degree_spec = st.selectbox("Degree Specialization", ["Science", "Commerce", "Arts", "Engineering"])
        internship = st.selectbox("Internship Experience", ["Yes", "No"])
        career_switch = st.selectbox("Career Switch Willingness", ["Yes", "No"])
        relevant_exp = st.selectbox("Relevant Experience", ["Yes", "No"])
        company_tier = st.selectbox("Company Tier", ["Tier 1", "Tier 2", "Tier 3"])
        job_match = st.selectbox("Job Role Match", ["Yes", "No"])
        competition = st.selectbox("Competition Level", ["Low", "Medium", "High"])
        bond = st.selectbox("Bond Requirement", ["Yes", "No"])
        layoff = st.selectbox("Layoff History", ["Yes", "No"])
        relocation = st.selectbox("Relocation Willingness", ["Willing", "Not Willing"])

    if st.button("🔮 Predict", type="primary"):
        from sklearn.preprocessing import LabelEncoder
        input_data = pd.DataFrame([{
            "age_years": age,
            "ssc_percentage": ssc,
            "hsc_percentage": hsc,
            "degree_percentage": degree,
            "technical_score": technical,
            "aptitude_score": aptitude,
            "communication_score": communication,
            "skills_match_percentage": skills,
            "certifications_count": certs,
            "years_of_experience": exp,
            "previous_ctc_lpa": prev_ctc,
            "expected_ctc_lpa": exp_ctc,
            "notice_period_days": notice,
            "employment_gap_months": emp_gap,
            "gender": gender,
            "degree_specialization": degree_spec,
            "internship_experience": internship,
            "career_switch_willingness": career_switch,
            "relevant_experience": relevant_exp,
            "company_tier": company_tier,
            "job_role_match": job_match,
            "competition_level": competition,
            "bond_requirement": bond,
            "layoff_history": layoff,
            "relocation_willingness": relocation
        }])

        # Encode categoricals
        le = LabelEncoder()
        for col in input_data.select_dtypes(include="object").columns:
            input_data[col] = le.fit_transform(input_data[col].astype(str))

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.markdown("---")
        if prediction == 1:
            st.success(f"✅ **PLACED** — Probability: {probability[1]*100:.1f}%")
        else:
            st.error(f"❌ **NOT PLACED** — Probability: {probability[0]*100:.1f}%")

        col1, col2 = st.columns(2)
        col1.metric("Placement Probability", f"{probability[1]*100:.1f}%")
        col2.metric("Rejection Probability", f"{probability[0]*100:.1f}%")

def tab_data(df):
    st.subheader("📋 Candidate Data")
    st.write(f"Total: {len(df):,} records")
    st.dataframe(df[["age_years","gender","Experience_Category",
                      "Academic_Band","Skills_Match_Level",
                      "Interview_Performance","company_tier","status"]
                   ].head(100), use_container_width=True)

def main():
    st.title("🎯 Job Acceptance Prediction System")
    st.markdown("*GUVI | HCL Capstone Project — HR Analytics & Machine Learning*")
    st.markdown("---")

    with st.spinner("Loading data..."):
        try:
            df = load_data()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    show_kpis(df)
    st.markdown("---")

    tab1,tab2,tab3,tab4 = st.tabs([
        "📊 EDA Analysis",
        "🤖 ML Models",
        "🔮 Predict",
        "📋 Data"
    ])
    with tab1: tab_eda(df)
    with tab2: tab_ml(df)
    with tab3: tab_predict(df)
    with tab4: tab_data(df)

if __name__ == "__main__":
    main()