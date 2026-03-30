import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD CLEANED DATA
# ─────────────────────────────────────────────
def load_data():
    print("[1] Loading cleaned data ...")
    df = pd.read_csv("data/candidates_cleaned.csv")
    print(f"    Shape: {df.shape}")
    return df

# ─────────────────────────────────────────────
# 2. PREPARE FEATURES
# ─────────────────────────────────────────────
def prepare_features(df):
    print("[2] Preparing features ...")

    # Select features for ML
    features = [
        "age_years", "ssc_percentage", "hsc_percentage",
        "degree_percentage", "technical_score", "aptitude_score",
        "communication_score", "skills_match_percentage",
        "certifications_count", "years_of_experience",
        "previous_ctc_lpa", "expected_ctc_lpa",
        "notice_period_days", "employment_gap_months",
        "gender", "degree_specialization", "internship_experience",
        "career_switch_willingness", "relevant_experience",
        "company_tier", "job_role_match", "competition_level",
        "bond_requirement", "layoff_history", "relocation_willingness"
    ]

    X = df[features].copy()
    y = df["target"]

    # Encode categorical columns
    cat_cols = X.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    print(f"    Features: {X.shape[1]}")
    print(f"    Target distribution:\n{y.value_counts()}")
    return X, y

# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X, y):
    print("\n[3] Training ML Models ...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # Model 1 - Logistic Regression
    print("\n    Training Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_acc = accuracy_score(y_test, lr_pred)
    results["Logistic Regression"] = lr_acc
    print(f"    Accuracy: {lr_acc:.4f}")

    # Model 2 - Random Forest
    print("\n    Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results["Random Forest"] = rf_acc
    print(f"    Accuracy: {rf_acc:.4f}")

    # Model 3 - Gradient Boosting
    print("\n    Training Gradient Boosting ...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    results["Gradient Boosting"] = gb_acc
    print(f"    Accuracy: {gb_acc:.4f}")

    # Best model
    best_name = max(results, key=results.get)
    print(f"\n    Best Model: {best_name} ({results[best_name]:.4f})")

    # Feature importance from Random Forest
    feature_imp = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)
    print(f"\n    Top 10 Features:\n{feature_imp.to_string(index=False)}")

    # Save best model
    print("\n[4] Saving best model ...")
    if best_name == "Random Forest":
        best_model = rf
        with open("data/best_model.pkl", "wb") as f:
            pickle.dump({"model": rf, "scaler": scaler,
                        "features": list(X.columns),
                        "model_name": best_name}, f)
    elif best_name == "Gradient Boosting":
        best_model = gb
        with open("data/best_model.pkl", "wb") as f:
            pickle.dump({"model": gb, "scaler": scaler,
                        "features": list(X.columns),
                        "model_name": best_name}, f)
    else:
        with open("data/best_model.pkl", "wb") as f:
            pickle.dump({"model": lr, "scaler": scaler,
                        "features": list(X.columns),
                        "model_name": best_name}, f)

    # Save results
    results_df = pd.DataFrame(list(results.items()),
                               columns=["Model", "Accuracy"])
    results_df.to_csv("data/model_results.csv", index=False)

    # Save feature importance
    feature_imp.to_csv("data/feature_importance.csv", index=False)

    print("    Model saved to data/best_model.pkl")
    print("\n" + "="*50)
    print("  MODEL TRAINING COMPLETE!")
    print("="*50)
    print(f"\n  Results Summary:")
    for model, acc in results.items():
        print(f"  {model:25s}: {acc*100:.2f}%")
    print(f"\n  Best: {best_name} ({results[best_name]*100:.2f}%)")

    return results, feature_imp

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("="*50)
    print("  Job Acceptance - ML Model Training")
    print("="*50)

    df = load_data()
    X, y = prepare_features(df)
    results, feature_imp = train_models(X, y)

if __name__ == "__main__":
    main()