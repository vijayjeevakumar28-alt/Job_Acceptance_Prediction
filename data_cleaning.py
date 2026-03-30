import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("[1] Loading data ...")
    df = pd.read_csv("data/HR_Job_Placement_Dataset.csv")
    print(f"    Shape: {df.shape}")
    return df

# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df):
    print("[2] Cleaning data ...")

    # Fill numeric missing values with median
    for col in ["ssc_percentage", "hsc_percentage",
                "notice_period_days", "employment_gap_months"]:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    for col in ["career_switch_willingness", "relevant_experience",
                "job_role_match", "layoff_history", "relocation_willingness"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Standardize text columns
    for col in ["gender", "degree_specialization", "internship_experience",
                "career_switch_willingness", "relevant_experience",
                "company_tier", "job_role_match", "competition_level",
                "bond_requirement", "layoff_history",
                "relocation_willingness", "status"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Clip invalid values
    df["ssc_percentage"]       = df["ssc_percentage"].clip(0, 100)
    df["hsc_percentage"]       = df["hsc_percentage"].clip(0, 100)
    df["degree_percentage"]    = df["degree_percentage"].clip(0, 100)
    df["technical_score"]      = df["technical_score"].clip(0, 100)
    df["aptitude_score"]       = df["aptitude_score"].clip(0, 100)
    df["communication_score"]  = df["communication_score"].clip(0, 100)
    df["skills_match_percentage"] = df["skills_match_percentage"].clip(0, 100)
    df["employment_gap_months"] = df["employment_gap_months"].clip(lower=0)
    df["notice_period_days"]   = df["notice_period_days"].clip(lower=0)

    print(f"    Missing values remaining: {df.isnull().sum().sum()}")
    return df

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df):
    print("[3] Adding derived features ...")

    # Experience Category
    def exp_category(y):
        if y == 0:    return "Fresher"
        if y <= 3:    return "Junior"
        if y <= 7:    return "Mid-Level"
        return "Senior"
    df["Experience_Category"] = df["years_of_experience"].apply(exp_category)

    # Academic Performance Band
    df["Academic_Score_Avg"] = (
        df["ssc_percentage"] +
        df["hsc_percentage"] +
        df["degree_percentage"]
    ) / 3

    def academic_band(score):
        if score >= 75: return "Excellent"
        if score >= 60: return "Good"
        if score >= 50: return "Average"
        return "Below Average"
    df["Academic_Band"] = df["Academic_Score_Avg"].apply(academic_band)

    # Skills Match Level
    def skills_level(s):
        if s >= 75: return "High"
        if s >= 50: return "Medium"
        return "Low"
    df["Skills_Match_Level"] = df["skills_match_percentage"].apply(skills_level)

    # Interview Performance
    df["Interview_Score_Avg"] = (
        df["technical_score"] +
        df["aptitude_score"] +
        df["communication_score"]
    ) / 3

    def interview_perf(score):
        if score >= 75: return "Excellent"
        if score >= 60: return "Good"
        if score >= 50: return "Average"
        return "Poor"
    df["Interview_Performance"] = df["Interview_Score_Avg"].apply(interview_perf)

    # Target variable: 1 = Placed, 0 = Not Placed
    df["target"] = (df["status"].str.lower() == "placed").astype(int)

    print(f"    Total columns now: {len(df.columns)}")
    return df

# ─────────────────────────────────────────────
# 4. STORE IN MYSQL
# ─────────────────────────────────────────────
def store_to_mysql(df):
    print("[4] Connecting to MySQL ...")
    try:
        pw = quote_plus("WelcomeGlobal1@")
        engine = create_engine(
            f"mysql+mysqlconnector://root:{pw}@127.0.0.1:3306/job_prediction_db"
        )
        df.to_sql("candidates", con=engine,
                  if_exists="replace", index=False, chunksize=500)
        print(f"    SUCCESS! {len(df)} rows inserted!")
        engine.dispose()
    except Exception as e:
        print(f"    ERROR: {e}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Job Acceptance Prediction - Data Cleaning")
    print("=" * 50)

    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    df.to_csv("data/candidates_cleaned.csv", index=False)
    print("    Cleaned CSV saved!")

    store_to_mysql(df)

if __name__ == "__main__":
    main()