import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

try:
    # xgboost is optional; handle gracefully if not installed
    from xgboost import XGBRegressor  # type: ignore
    XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    XGB_AVAILABLE = False


st.set_page_config(page_title="AI Job Displacement Explorer", page_icon="🧠", layout="wide")
sns.set_theme(style="whitegrid")


def load_default_csv_if_present() -> Tuple[pd.DataFrame | None, str | None]:
    candidate_names = [
        "ai_job_market_insights.csv",
    ]
    script_dir = os.path.dirname(__file__)
    for name in candidate_names:
        candidate_path = os.path.join(script_dir, name)
        if os.path.exists(candidate_path):
            try:
                return pd.read_csv(candidate_path), candidate_path
            except Exception:
                pass
    return None, None


def recommend_safe_jobs(df: pd.DataFrame) -> pd.DataFrame:
    risk_score_map: Dict[str, int] = {"Low": 3, "Medium": 2, "High": 1}
    growth_score_map: Dict[str, int] = {"Decline": 1, "Stable": 2, "Growth": 3}

    work = df.copy()
    work["Risk_Score"] = work["Automation_Risk"].map(risk_score_map)
    work["Growth_Score"] = work["Job_Growth_Projection"].map(growth_score_map)
    work["Stability_Score"] = (work["Risk_Score"] * 0.6) + (work["Growth_Score"] * 0.4)

    cols = [
        "Job_Title",
        "Industry",
        "Salary_USD",
        "Automation_Risk",
        "Job_Growth_Projection",
        "AI_Adoption_Level",
        "Remote_Friendly",
    ]
    cols = [c for c in cols if c in work.columns]
    top = (
        work.sort_values(["Stability_Score", "Salary_USD"], ascending=[False, False])
        .loc[:, cols]
        .head(10)
        .reset_index(drop=True)
    )
    return top


def encode_and_cluster(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    encoded = df.copy()
    categorical_cols = [
        "Job_Title",
        "Industry",
        "Company_Size",
        "Location",
        "AI_Adoption_Level",
        "Automation_Risk",
        "Required_Skills",
        "Remote_Friendly",
        "Job_Growth_Projection",
    ]
    for col in categorical_cols:
        if col in encoded.columns:
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col].astype(str))

    numeric_features = encoded.select_dtypes(include=["int64", "float64"])  # after label encoding
    if len(numeric_features.columns) == 0:
        encoded["Cluster"] = 0
    else:
        n_clusters = 3 if len(numeric_features) >= 3 else max(1, len(numeric_features))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        encoded["Cluster"] = km.fit_predict(numeric_features)
    return encoded, numeric_features


def train_models(df_encoded: pd.DataFrame) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    feature_cols = [
        "Industry",
        "Company_Size",
        "Location",
        "AI_Adoption_Level",
        "Automation_Risk",
        "Remote_Friendly",
        "Job_Growth_Projection",
        "Cluster",
    ]
    feature_cols = [c for c in feature_cols if c in df_encoded.columns]
    if "Salary_USD" not in df_encoded.columns or not feature_cols:
        return {}, pd.DataFrame()

    X = df_encoded[feature_cols]
    y = df_encoded["Salary_USD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: Dict[str, Tuple[float, float]] = {}

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results["Linear Regression"] = (
        float(r2_score(y_test, y_pred_lr)),
        float(mean_absolute_error(y_test, y_pred_lr)),
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = (
        float(r2_score(y_test, y_pred_rf)),
        float(mean_absolute_error(y_test, y_pred_rf)),
    )

    if XGB_AVAILABLE:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        results["XGBoost"] = (
            float(r2_score(y_test, y_pred_xgb)),
            float(mean_absolute_error(y_test, y_pred_xgb)),
        )

    comparison_df = (
        pd.DataFrame(
            [
                {"Model": name, "R2": m[0], "MAE": m[1]}
                for name, m in results.items()
            ]
        )
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )
    return results, comparison_df


def main() -> None:
    st.title("🧠 Forecasting AI-Induced Job Displacement")
    st.caption("Explore job market data, visualize trends, and compare salary prediction models.")

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
        df: pd.DataFrame | None = None
        source_info = ""
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                source_info = "(from upload)"
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
        else:
            default_df, default_path = load_default_csv_if_present()
            if default_df is not None:
                df = default_df
                source_info = f"(auto-loaded: {os.path.basename(default_path)})"

        st.markdown("---")
        st.header("Options")
        sample_n = st.slider("Preview rows", min_value=5, max_value=50, value=10, step=5)

    if df is None:
        st.info(
            "Upload a dataset with columns like `Job_Title`, `Industry`, `Salary_USD`,\n"
            "`Automation_Risk`, and `Job_Growth_Projection` to get started."
        )
        return

    st.subheader("Dataset Preview " + source_info)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df.head(sample_n), use_container_width=True)
    with c2:
        st.write("Shape", df.shape)
        st.write("Dtypes")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

    # Tabs for analysis
    t_overview, t_univariate, t_bivariate, t_corr, t_models, t_reco = st.tabs(
        [
            "Overview",
            "Univariate",
            "Bivariate",
            "Correlation",
            "Models",
            "Recommendations",
        ]
    )

    with t_overview:
        st.markdown("### Summary Stats")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    with t_univariate:
        st.markdown("### Numerical Distributions")
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if num_cols:
            cols = st.multiselect("Choose numeric columns", options=num_cols, default=num_cols)
            for col in cols:
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.histplot(df[col].dropna(), kde=True, color="teal", ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("No numeric columns found.")

        st.markdown("### Categorical Frequencies")
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            cols = st.multiselect("Choose categorical columns", options=cat_cols, default=cat_cols[:5])
            top_k = st.slider("Show top K categories", 3, 20, 10)
            for col in cols:
                fig, ax = plt.subplots(figsize=(6, 3))
                df[col].value_counts().head(top_k).plot(kind="bar", color="skyblue", ax=ax)
                ax.set_title(f"Frequency of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    with t_bivariate:
        st.markdown("### Salary vs Required Skills (Violin)")
        if "Required_Skills" in df.columns and "Salary_USD" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.violinplot(data=df, x="Required_Skills", y="Salary_USD", ax=ax)
            ax.set_title("Salary Distribution Across Required Skills")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Columns `Required_Skills` and/or `Salary_USD` not found.")

        st.markdown("### Mean Salary by Job Title (Bar)")
        if "Job_Title" in df.columns and "Salary_USD" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=df, x="Job_Title", y="Salary_USD", errorbar="sd", ax=ax)
            ax.set_title("Mean Salary by Job Title")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Columns `Job_Title` and/or `Salary_USD` not found.")

    with t_corr:
        st.markdown("### Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        if corr.size > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap of Numerical Features")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No numeric features available for correlation.")

    with t_models:
        st.markdown("### Feature Encoding, Clustering, and Model Comparison")
        with st.spinner("Encoding features and clustering..."):
            encoded, _ = encode_and_cluster(df)

        with st.spinner("Training models..."):
            results, comp_df = train_models(encoded)

        if not results:
            st.warning("Insufficient columns to train models. Ensure `Salary_USD` and feature columns exist.")
        else:
            st.write("Scores (higher R² is better, lower MAE is better):")
            st.dataframe(comp_df, use_container_width=True)

            fig1, ax1 = plt.subplots(figsize=(6, 3))
            sns.barplot(data=comp_df, x="Model", y="R2", ax=ax1)
            ax1.set_title("Model Comparison (R²)")
            plt.xticks(rotation=0)
            st.pyplot(fig1, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(6, 3))
            sns.barplot(data=comp_df, x="Model", y="MAE", ax=ax2)
            ax2.set_title("Model Comparison (MAE)")
            plt.xticks(rotation=0)
            st.pyplot(fig2, use_container_width=True)

            if not XGB_AVAILABLE:
                st.caption("XGBoost not installed; install `xgboost` to include it in comparisons.")

    with t_reco:
        st.markdown("### Top 10 Future-Safe Jobs")
        required_cols = {"Automation_Risk", "Job_Growth_Projection", "Salary_USD"}
        if required_cols.issubset(df.columns):
            top_jobs = recommend_safe_jobs(df)
            st.dataframe(top_jobs, use_container_width=True)

            if {"Job_Title", "Salary_USD", "Automation_Risk"}.issubset(top_jobs.columns):
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    data=top_jobs,
                    x="Job_Title",
                    y="Salary_USD",
                    hue="Automation_Risk",
                    ax=ax,
                )
                ax.set_title("Top 10 Future-Safe Jobs by Salary & Automation Risk")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Dataset missing required columns for recommendations.")


if __name__ == "__main__":
    main()


