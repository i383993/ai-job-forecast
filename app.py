import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuration and Data Loading ---

# Assuming 'ai_job_market_insights.csv' is available. 
# It is assumed to have columns: 
# 'Job_Title', 'Salary_USD', 'Industry', 'Automation_Risk', 'AI_Adoption_Level'
try:
    df = pd.read_csv("ai_job_market_insights.csv")
    
    # Simple Data Pre-processing for robust visualization
    # Ensure numerical columns are correctly typed
    if 'Salary_USD' in df.columns:
        df['Salary_USD'] = pd.to_numeric(df['Salary_USD'], errors='coerce')
    
    # Map Automation Risk to readable strings if stored as numbers (0, 1, 2)
    risk_map = {0: 'Low/None', 1: 'Moderate', 2: 'High/Advanced'}
    if df['Automation_Risk'].dtype in ['int64', 'float64']:
        df['Automation_Risk_Label'] = df['Automation_Risk'].map(risk_map).fillna('Unknown')
    else:
        df['Automation_Risk_Label'] = df['Automation_Risk'] # Assume it's already a string

    data_loaded = True
except FileNotFoundError:
    st.error("Error: 'ai_job_market_insights.csv' not found. Please ensure the data file is in the same directory.")
    data_loaded = False
except Exception as e:
    st.error(f"An error occurred during data loading or processing: {e}")
    data_loaded = False

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="AI Career Navigator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Career Navigator: Market Insights & Deep Dives")
st.markdown("Explore macro market trends or dive deep into the risk and salary profiles of specific job titles.")

if data_loaded:
    # --- Main Navigation ---
    analysis_mode = st.radio(
        "Select Analysis Mode",
        ('Industry Analysis', 'Job Title Deep Dive'),
        horizontal=True,
        index=0 # Default to Industry Analysis
    )

    # --- Section 1: Industry Analysis (Based on user's original code) ---
    if analysis_mode == 'Industry Analysis':
        st.header("1. Macro-Level Industry Analysis")
        
        # Filters
        industries = df["Industry"].unique()
        selected_industry = st.selectbox(
            "Select Industry to Filter By", 
            industries, 
            key="industry_select"
        )

        filtered_df = df[df["Industry"] == selected_industry]
        
        if filtered_df.empty:
            st.warning(f"No data available for the selected industry: {selected_industry}")
        else:
            # Visualization 1: Salary Distribution (Box Plot)
            fig_salary = px.box(
                filtered_df, 
                y="Salary_USD", 
                title=f"Salary Distribution Across All Jobs in {selected_industry}",
                labels={'Salary_USD': 'Annual Salary (USD)'},
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_salary, use_container_width=True)

            # Visualization 2: Automation Risk (Histogram for all industries)
            # This is better done on the full dataset to show relative distribution
            fig_auto = px.histogram(
                df, 
                x="Automation_Risk_Label", 
                color="AI_Adoption_Level",
                title="Automation Risk Distribution by AI Adoption Level (Global)",
                labels={'Automation_Risk_Label': 'Automation Risk Level'},
                category_orders={"Automation_Risk_Label": ['Low/None', 'Moderate', 'High/Advanced']},
                color_discrete_sequence=px.colors.qualitative.G10
            )
            st.plotly_chart(fig_auto, use_container_width=True)

            # Simple Forecasting Insight
            st.subheader("📈 AI Impact Insight")
            risk_levels = filtered_df["Automation_Risk_Label"].unique()
            if 'High/Advanced' in risk_levels:
                st.warning(f"🚨 **High automation risk** detected among some job roles in {selected_industry} — potential job displacement likely without upskilling.")
            else:
                st.success(f"✅ {selected_industry} currently shows stable or growing job trends relative to AI displacement risk.")

            # Summary Stats
            st.subheader("📊 Summary Statistics for Selected Industry")
            st.dataframe(
                filtered_df[['Salary_USD', 'Automation_Risk']].describe().round(0)
            )


    # --- Section 2: Job Title Deep Dive ---
    elif analysis_mode == 'Job Title Deep Dive':
        st.header("2. Detailed Job Title Profile")

        # Filters for Job Title
        job_titles = sorted(df["Job_Title"].unique())
        selected_job = st.selectbox(
            "Select a Job Title for Deep Dive Analysis", 
            job_titles, 
            key="job_select"
        )
        
        job_df = df[df["Job_Title"] == selected_job]

        if job_df.empty:
            st.warning(f"No data available for the selected job title: {selected_job}")
        else:
            col1, col2 = st.columns(2)

            with col1:
                # Visualization A: Salary Distribution for the Selected Job
                fig_job_salary = px.histogram(
                    job_df, 
                    x="Salary_USD", 
                    marginal="box", # Adds a box plot for summary
                    title=f"Salary Distribution for: {selected_job}",
                    labels={'Salary_USD': 'Annual Salary (USD)'},
                    color_discrete_sequence=['#4CAF50'] # Green
                )
                st.plotly_chart(fig_job_salary, use_container_width=True)

            with col2:
                # Visualization B: Risk and Average Salary Summary
                avg_salary = job_df['Salary_USD'].mean()
                risk_level = job_df['Automation_Risk_Label'].mode()[0]
                top_industries = job_df['Industry'].value_counts().head(3).index.tolist()

                st.subheader(f"Key Insights for {selected_job}")
                
                # Metric 1: Average Salary
                st.metric(
                    label="Average Reported Salary",
                    value=f"${avg_salary:,.0f}",
                    delta_color="off"
                )
                
                # Metric 2: Automation Risk
                st.markdown(f"**Automation Risk:**")
                if risk_level == 'High/Advanced':
                    st.error(f"**{risk_level}** - High vulnerability to AI displacement.")
                elif risk_level == 'Moderate':
                    st.warning(f"**{risk_level}** - Some tasks are automatable; continuous upskilling is advised.")
                else:
                    st.success(f"**{risk_level}** - Stable outlook, often requiring high-level human judgment.")
                
                # Metric 3: Top Industries
                st.markdown(f"**Top Industries:** {', '.join(top_industries)}")
                
            st.markdown("---")

            # Visualization C: Salary vs AI Adoption (Scatter/Jitter Plot)
            # Shows how salaries for this job vary based on industry AI Adoption levels
            fig_adoption_salary = px.strip(
                job_df,
                x="AI_Adoption_Level",
                y="Salary_USD",
                color="Industry",
                title=f"Salary Spread by AI Adoption Level for {selected_job}",
                labels={'Salary_USD': 'Annual Salary (USD)', 'AI_Adoption_Level': 'Industry AI Adoption'},
                stripmode='overlay',
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            st.plotly_chart(fig_adoption_salary, use_container_width=True)


st.caption("Developed by Isaiah Panicker • Data Science Project")

# =========================================================================
# --- SECTION 4: Top 10 Future-Safe Job Recommendations ---
# =========================================================================

elif mode == "4. Future-Safe Job Recommendations":
    st.header("4. Top 10 Future-Safe Job Recommendations")
    st.markdown("This section identifies the jobs least likely to be replaced by AI, based on automation risk and skill-growth balance.")

    # Check if essential columns exist
    if 'Automation_Risk' not in df.columns or 'Salary_USD' not in df.columns:
        st.error("⚠️ Required columns ('Automation_Risk' and 'Salary_USD') not found in dataset.")
    else:
        # Add optional mapping for growth or skill progression if available
        if 'Job_Growth_Projection' in df.columns:
            growth_map = {'Decline': 1, 'Stable': 2, 'Growth': 3, 'High Growth': 4}
            df['Growth_Score'] = df['Job_Growth_Projection'].map(growth_map).fillna(2)
        else:
            df['Growth_Score'] = 2  # neutral default

        risk_map = {'Low/None': 3, 'Moderate': 2, 'High/Advanced': 1}
        df['Risk_Score'] = df['Automation_Risk'].map(risk_map).fillna(1)

        # Composite stability metric
        df['Stability_Score'] = (df['Risk_Score'] * 0.6) + (df['Growth_Score'] * 0.4)

        # Sort and select top 10
        top_jobs = df.sort_values(['Stability_Score', 'Salary_USD'], ascending=[False, False]).head(10)

        st.success("✅ Top 10 jobs that are both high-paying and resistant to automation:")

        # Display top jobs in a table
        display_cols = ['Job_Title', 'Industry', 'Salary_USD', 'Automation_Risk']
        if 'Job_Growth_Projection' in df.columns:
            display_cols.append('Job_Growth_Projection')
        if 'AI_Adoption_Level' in df.columns:
            display_cols.append('AI_Adoption_Level')

        st.dataframe(top_jobs[display_cols].reset_index(drop=True))

        # Visualization
        import plotly.express as px
        fig = px.bar(
            top_jobs,
            x='Job_Title',
            y='Salary_USD',
            color='Automation_Risk',
            title="💼 Top 10 Future-Safe Jobs by Salary & Automation Risk",
            hover_data=display_cols,
            color_discrete_map={'Low/None': 'green', 'Moderate': 'orange', 'High/Advanced': 'red'}
        )
        fig.update_layout(xaxis_title="Job Title", yaxis_title="Salary (USD)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Optional insights
        st.markdown("### 🧭 Insight")
        st.info("These jobs show a balance between high salaries and low automation risk, indicating strong long-term viability in the AI-driven future.")
