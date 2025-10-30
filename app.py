import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
import plotly.express as px

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
    t_overview, t_univariate, t_bivariate, t_corr, t_models, t_reco, t_nav, t_geo = st.tabs(
        [
            "Overview",
            "Univariate",
            "Bivariate",
            "Correlation",
            "Models",
            "Recommendations",
            "AI Career Navigator",
            "Geography",
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


    # Plotly-powered navigator views from user's request
    with t_nav:
        st.markdown("### 🤖 AI Career Navigator: Market Insights & Deep Dives")
        st.markdown("Explore macro market trends or dive deep into the risk and salary profiles of specific job titles.")

        # Light preprocessing for robustness
        work = df.copy()
        if "Salary_USD" in work.columns:
            work["Salary_USD"] = pd.to_numeric(work["Salary_USD"], errors="coerce")
        risk_map = {0: "Low/None", 1: "Moderate", 2: "High/Advanced"}
        if "Automation_Risk" in work.columns:
            if str(work["Automation_Risk"].dtype) in ["int64", "float64"]:
                work["Automation_Risk_Label"] = work["Automation_Risk"].map(risk_map).fillna("Unknown")
            else:
                work["Automation_Risk_Label"] = work["Automation_Risk"].astype(str)

        mode = st.radio(
            "Select Analysis Mode",
            ("Industry Analysis", "Job Title Deep Dive"),
            horizontal=True,
            index=0,
        )

        if mode == "Industry Analysis":
            st.subheader("1. Macro-Level Industry Analysis")
            if "Industry" not in work.columns:
                st.warning("Column `Industry` not found in dataset.")
            else:
                industries = sorted(work["Industry"].dropna().unique().tolist())
                selected_industry = st.selectbox("Select Industry to Filter By", industries, key="industry_select")
                filtered = work[work["Industry"] == selected_industry]

                if filtered.empty:
                    st.warning(f"No data available for the selected industry: {selected_industry}")
                else:
                    # Salary distribution (box)
                    if "Salary_USD" in filtered.columns:
                        fig_salary = px.box(
                            filtered,
                            y="Salary_USD",
                            title=f"Salary Distribution Across All Jobs in {selected_industry}",
                            labels={"Salary_USD": "Annual Salary (USD)"},
                            color_discrete_sequence=px.colors.qualitative.Plotly,
                        )
                        st.plotly_chart(fig_salary, use_container_width=True)

                    # Automation risk distribution (global)
                    if {"Automation_Risk_Label", "AI_Adoption_Level"}.issubset(work.columns):
                        fig_auto = px.histogram(
                            work,
                            x="Automation_Risk_Label",
                            color="AI_Adoption_Level",
                            title="Automation Risk Distribution by AI Adoption Level (Global)",
                            labels={"Automation_Risk_Label": "Automation Risk Level"},
                            category_orders={"Automation_Risk_Label": ["Low/None", "Moderate", "High/Advanced"]},
                            color_discrete_sequence=px.colors.qualitative.G10,
                        )
                        st.plotly_chart(fig_auto, use_container_width=True)

                    # Insight
                    st.subheader("📈 AI Impact Insight")
                    risk_levels = filtered.get("Automation_Risk_Label", pd.Series(dtype=str)).unique().tolist()
                    if "High/Advanced" in risk_levels:
                        st.warning(
                            f"🚨 High automation risk detected among some job roles in {selected_industry} — potential job displacement likely without upskilling."
                        )
                    else:
                        st.success(
                            f"✅ {selected_industry} currently shows stable or growing job trends relative to AI displacement risk."
                        )

                    # Summary stats
                    if {"Salary_USD", "Automation_Risk"}.issubset(filtered.columns):
                        st.subheader("📊 Summary Statistics for Selected Industry")
                        st.dataframe(filtered[["Salary_USD", "Automation_Risk"]].describe().round(0))

        else:  # Job Title Deep Dive
            st.subheader("2. Detailed Job Title Profile")
            if "Job_Title" not in work.columns:
                st.warning("Column `Job_Title` not found in dataset.")
            else:
                job_titles = sorted(work["Job_Title"].dropna().unique().tolist())
                selected_job = st.selectbox("Select a Job Title for Deep Dive Analysis", job_titles, key="job_select")
                job_df = work[work["Job_Title"] == selected_job]

                if job_df.empty:
                    st.warning(f"No data available for the selected job title: {selected_job}")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        if "Salary_USD" in job_df.columns:
                            fig_job_salary = px.histogram(
                                job_df,
                                x="Salary_USD",
                                marginal="box",
                                title=f"Salary Distribution for: {selected_job}",
                                labels={"Salary_USD": "Annual Salary (USD)"},
                                color_discrete_sequence=["#4CAF50"],
                            )
                            st.plotly_chart(fig_job_salary, use_container_width=True)
                    with col2:
                        avg_salary = float(job_df["Salary_USD"].mean()) if "Salary_USD" in job_df.columns else float("nan")
                        risk_level = (
                            job_df.get("Automation_Risk_Label", pd.Series(["Unknown"])) .mode()[0]
                            if "Automation_Risk_Label" in job_df.columns and not job_df["Automation_Risk_Label"].empty
                            else "Unknown"
                        )
                        top_industries = (
                            job_df["Industry"].value_counts().head(3).index.tolist()
                            if "Industry" in job_df.columns else []
                        )
                        st.subheader(f"Key Insights for {selected_job}")
                        if not np.isnan(avg_salary):
                            st.metric(label="Average Reported Salary", value=f"${avg_salary:,.0f}", delta_color="off")
                        st.markdown("**Automation Risk:**")
                        if risk_level == "High/Advanced":
                            st.error(f"**{risk_level}** - High vulnerability to AI displacement.")
                        elif risk_level == "Moderate":
                            st.warning(f"**{risk_level}** - Some tasks are automatable; continuous upskilling is advised.")
                        else:
                            st.success(f"**{risk_level}** - Stable outlook, often requiring high-level human judgment.")
                        if top_industries:
                            st.markdown(f"**Top Industries:** {', '.join(top_industries)}")

                    st.markdown("---")

                    if {"AI_Adoption_Level", "Salary_USD"}.issubset(job_df.columns):
                        fig_adoption_salary = px.strip(
                            job_df,
                            x="AI_Adoption_Level",
                            y="Salary_USD",
                            color="Industry" if "Industry" in job_df.columns else None,
                            title=f"Salary Spread by AI Adoption Level for {selected_job}",
                            labels={"Salary_USD": "Annual Salary (USD)", "AI_Adoption_Level": "Industry AI Adoption"},
                            stripmode="overlay",
                            color_discrete_sequence=px.colors.qualitative.Dark24,
                        )
                        st.plotly_chart(fig_adoption_salary, use_container_width=True)

        st.caption("Developed by Isaiah Panicker • Data Science Project")

    with t_geo:
        st.markdown("### 🌍 Jobs by Country/Location")
        # Prefer a 'Country' column if present; fall back to 'Location'
        geo_col = "Country" if "Country" in df.columns else ("Location" if "Location" in df.columns else None)
        if geo_col is None:
            st.info("No `Country` or `Location` column found for geographic visualization.")
        else:
            counts = df[geo_col].astype(str).value_counts().reset_index()
            counts.columns = [geo_col, "Jobs"]
            st.subheader("Jobs count by location")
            fig_bar = px.bar(counts.head(30), x=geo_col, y="Jobs", title="Top Locations by Job Count")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Choropleth only if we have a Country column (names likely to match)
            if geo_col == "Country":
                try:
                    fig_map = px.choropleth(
                        counts,
                        locations="Country",
                        locationmode="country names",
                        color="Jobs",
                        color_continuous_scale="Blues",
                        title="Jobs per Country",
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception:
                    st.caption("Could not render world map; showing bar chart instead.")


if __name__ == "__main__":
    main()


