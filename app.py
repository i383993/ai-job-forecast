import streamlit as st
import pandas as pd
import plotly.express as px

# --- Load Data ---
df = pd.read_csv("ai_job_market_insights.csv")

st.title("🤖 Forecasting AI-Induced Job Displacement")
st.markdown("Analyze how automation and AI adoption impact job trends across industries.")

# --- Show Raw Data ---
if st.checkbox("Show Dataset"):
    st.write(df.head())

# --- Filters ---
industries = df["Industry"].unique()
selected_industry = st.selectbox("Select Industry", industries)

filtered_df = df[df["Industry"] == selected_industry]

# --- Visualization 1: Salary Distribution ---
fig_salary = px.box(filtered_df, y="Salary_USD", title=f"Salary Distribution in {selected_industry}")
st.plotly_chart(fig_salary)

# --- Visualization 2: Automation Risk ---
fig_auto = px.histogram(df, x="Automation_Risk", color="AI_Adoption_Level",
                        title="Automation Risk vs AI Adoption")
st.plotly_chart(fig_auto)

# --- Simple Forecasting Insight ---
st.subheader("📈 AI Impact Insight")
if "High" in filtered_df["Automation_Risk"].values:
    st.warning(f"High automation risk detected in {selected_industry} — potential job displacement likely.")
else:
    st.success(f"{selected_industry} currently shows stable or growing job trends.")

# --- Summary Stats ---
st.subheader("📊 Summary Statistics")
st.write(filtered_df.describe())

st.caption("Developed by Isaiah Panicker • Forecasting AI-Induced Job Displacement")
