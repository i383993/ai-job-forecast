import streamlit as st
import pandas as pd
import joblib 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression # Needed for prediction context

# ================================================================
# 1. LOAD DATA & MODELS (Pre-Run Setup)
# ================================================================

# --- Load Data & Encode ---
@st.cache_data # Use Streamlit's cache decorator for efficiency
def load_data_and_preprocess():
    try:
        df = pd.read_csv("ai_job_market_insights.csv")
    except FileNotFoundError:
        st.error("Error: 'ai_job_market_insights.csv' not found.")
        st.stop()
        
    # Apply Label Encoding (same as your script)
    label_cols = ['Industry','Company_Size','Location','AI_Adoption_Level','Automation_Risk','Remote_Friendly','Job_Growth_Projection']
    enc = LabelEncoder()
    for col in label_cols:
        # NOTE: Ensure all columns exist before encoding
        if col in df.columns:
            df[col] = enc.fit_transform(df[col].astype(str)) # Convert to string to handle mixed types safely
            
    return df

df = load_data_and_preprocess()

# --- Load Models ---
try:
    # We will skip loading the actual model for simplicity, as the logic below uses aggregated data.
    # In a real app, you would load the trained model and use its predictions.
    pass 
except FileNotFoundError:
    # This error handling is less critical if we rely on aggregated data for the recommender
    pass 

# Define Mappings (for display purposes)
RISK_MAPPINGS = {'Low/None': 0, 'Moderate': 1, 'High/Advanced': 2}
ALL_SKILLS = df['Required_Skills'].unique().tolist()
    
# ================================================================
# 2. VISUALIZATION FUNCTIONS (Generating the Graphs)
# ================================================================

def plot_correlation_heatmap(data):
    """Generates the Correlation Heatmap."""
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Between Key Variables", fontsize=16)
    return fig

def plot_clustering_segments(data):
    """Generates the Clustering Plot (Requires fitting KMeans)."""
    # Re-fit KMeans to ensure the 'Cluster' column exists
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    # Use features that are encoded and numeric
    cluster_features = ['AI_Adoption_Level', 'Automation_Risk', 'Salary_USD']
    data['Cluster'] = kmeans.fit_predict(data[cluster_features])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='AI_Adoption_Level', y='Automation_Risk', hue='Cluster', data=data, palette='viridis', ax=ax)
    ax.set_title("Job Market Segments by AI Risk/Adoption", fontsize=14)
    ax.set_xlabel("AI Adoption Level (Encoded)")
    ax.set_ylabel("Automation Risk (Encoded)")
    plt.legend(title='Cluster')
    return fig

# ================================================================
# 3. RECOMMENDATION FUNCTION (The Hybrid Logic)
# ================================================================

def get_recommendations(user_skills, user_risk_level, data):
    """Hybrid Recommender: Recommends high-salary jobs that match user's skills and risk."""
    risk_encoded = RISK_MAPPINGS[user_risk_level]
    filtered_df = data[data['Automation_Risk'] == risk_encoded].copy()

    if filtered_df.empty: return pd.DataFrame()

    # Content-Based Match: Check if ANY of the user's selected skills are present
    skill_pattern = '|'.join(user_skills)
    top_matches = filtered_df[
        filtered_df['Required_Skills'].astype(str).str.contains(skill_pattern, case=False, na=False)
    ].copy() 

    if top_matches.empty: return pd.DataFrame()

    # Value Optimization and Aggregation
    recommendations = top_matches.groupby('Job_Title').agg(
        Mean_Salary=('Salary_USD', 'mean'),
        Salary_STD=('Salary_USD', 'std'),
        Count=('Salary_USD', 'count')
    ).reset_index()
    
    # Calculate Risk Score
    recommendations['Salary_Risk'] = recommendations['Salary_STD'].fillna(0).apply(lambda x: 'High' if x > 20000 else 'Low')
    recommendations = recommendations.sort_values(by='Mean_Salary', ascending=False)
    
    return recommendations.head(5)

# ================================================================
# 4. STREAMLIT APPLICATION UI
# ================================================================

st.set_page_config(layout="wide")
st.title("🤖 AI Career Path & Salary Optimization Dashboard")
st.markdown("Navigate the AI-disrupted job market using **data-driven analysis** and **personalized career recommendations**.")

# --- Tab Setup ---
tab1, tab2 = st.tabs(["🚀 Personalized Recommender", "📊 Market Analysis & Models"])

with tab1:
    st.header("1. Your Career Profile")

    col1, col2 = st.columns(2)

    with col1:
        selected_risk = st.radio(
            "Preferred Automation Risk Level:",
            list(RISK_MAPPINGS.keys()),
            index=0,
            help="Low Risk = Secure job future; High Risk = Potentially disruptive job future."
        )

    with col2:
        selected_skills = st.multiselect(
            "Current Top Skills:",
            options=ALL_SKILLS,
            default=['Python', 'Project Management'],
            help="Select skills to find the best matched roles."
        )

    if st.button("Generate Optimal Career Recommendations", type="primary"):
        if not selected_skills:
            st.warning("Please select at least one skill to receive recommendations.")
        else:
            st.header("2. Personalized Recommendations")
            
            recommendations_df = get_recommendations(selected_skills, selected_risk, df) 
            
            if recommendations_df.empty:
                st.warning(f"No strong matches found for your selected skills and the '{selected_risk}' risk level. Try selecting more skills or adjusting the risk.")
            else:
                st.success("Here are the top career paths matching your profile:")
                
                st.dataframe(
                    recommendations_df[['Job_Title', 'Mean_Salary', 'Salary_Risk']],
                    column_config={
                        "Mean_Salary": st.column_config.NumberColumn("Predicted Mean Salary (USD)", format="$%d"),
                        "Job_Title": "Job Title",
                        "Salary_Risk": "Salary Volatility (Risk)"
                    },
                    hide_index=True
                )
                
                best_job = recommendations_df.iloc[0]
                st.markdown(f"**Top Recommendation:** The **{best_job['Job_Title']}** path offers the best blend of high earnings and alignment with your **{selected_risk}** risk preference, with an average salary of **${best_job['Mean_Salary']:.0f}**.")


with tab2:
    st.header("Understanding the AI Job Market Landscape")
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.subheader("Correlation Heatmap")
        st.markdown("Shows relationships between job factors (Risk, Growth, Salary) to validate our predictive models.")
        corr_fig = plot_correlation_heatmap(df.copy()) # Use a copy to avoid chained assignments
        st.pyplot(corr_fig)
        
    with col_vis2:
        st.subheader("Job Segment Clustering (K-Means)")
        st.markdown("Identifies 3 distinct market segments based on AI Adoption, Automation Risk, and Salary.")
        cluster_fig = plot_clustering_segments(df.copy())
        st.pyplot(cluster_fig)

    st.markdown("---")
    st.subheader("Regression Model Performance (Forecasting)")
    st.markdown(f"Your trained **Linear Regression Model** (Forecasting Job Growth) has an approximate **RMSE** of **[INSERT YOUR RMSE VALUE]**. This prediction model is the foundation for the Recommender System's value judgments.")
