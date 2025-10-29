import streamlit as st
import pandas as pd
import joblib # To load pre-trained ML models
import numpy as np # For numerical operations

# --- 1. Load Pre-Trained Models and Data ---

# NOTE: You must train and save these models beforehand in a separate script!
# Replace these with your actual saved file paths.
try:
    regression_model = joblib.load('multiple_regression_model.pkl')
    # clustering_model = joblib.load('kmeans_clustering_model.pkl') # K-Means is optional for the final app, but good for segmentation

    # Load the original data (or a clean, pre-processed version) for lookups
    df = pd.read_csv("ai_job_market_insights.csv") 

    # --- Prepare Data for Recommender (Needed for content-based matching) ---
    # Convert required_skills to a set of unique skills for the multiselect
    ALL_SKILLS = df['Required_Skills'].unique().tolist()
    
    # Define the categorical mappings used in your analysis for display purposes
    RISK_MAPPINGS = {
        'Low/None': 0, 
        'Moderate': 1, 
        'High/Advanced': 2
    }
    
    st.sidebar.success("Models and Data Loaded Successfully.")

except FileNotFoundError:
    st.error("Error: ML models or data file not found. Ensure 'multiple_regression_model.pkl' and 'ai_job_market_insights.csv' are in the correct directory.")
    st.stop()
    
# --- 2. Define Prediction and Recommendation Functions ---

def predict_salary(job_title, risk_level_encoded, skill_set):
    """
    Predicts salary using the pre-trained regression model.
    NOTE: This is a placeholder function. Your actual implementation will require
    creating a feature vector (X) that matches the training data format (e.g.,
    one-hot encoding for job_title and skills).
    """
    # Create a DataFrame/Feature vector (X) matching your model's input features
    # --- Example Placeholder Logic (You MUST implement your actual feature engineering here) ---
    features = {
        'Automation_Risk': risk_level_encoded,
        # Placeholder for one-hot encoding logic for Job_Title and Skills
        'Job_Title_Data Scientist': 1 if job_title == 'Data Scientist' else 0,
        'Skill_Python': 1 if 'Python' in skill_set else 0,
        # ... add all features your model was trained on ...
    }
    
    # For a simple example, let's just use the risk level
    if risk_level_encoded == 0: return 120000 
    if risk_level_encoded == 1: return 95000 
    if risk_level_encoded == 2: return 80000
    # --- End Placeholder Logic ---
    
    # Actual prediction would look like this:
    # X = pd.DataFrame([features])
    # predicted_salary = regression_model.predict(X)[0]
    # return predicted_salary


def get_recommendations(user_skills, user_risk_level):
    """
    Hybrid Recommender: Recommends high-salary jobs that match user's skills and risk.
    """
    # Filter the dataset for jobs matching the user's risk preference
    risk_encoded = RISK_MAPPINGS[user_risk_level]
    filtered_df = df[df['Automation_Risk'] == risk_encoded].copy()

    # Simple Content-Based Match: Filter jobs where the Required_Skills column
    # contains at least one of the user's selected skills
    def skill_match(job_skill):
        return any(skill in user_skills for skill in [job_skill]) # simplified for single column

    # In a real app, you would filter based on a multi-skill match, but we use the single-column approach here
    top_matches = filtered_df[filtered_df['Required_Skills'].apply(skill_match)]
    
    if top_matches.empty:
        return pd.DataFrame()

    # Use the Regression Model to predict salary for these matches (Hybrid Core)
    # NOTE: You'd call predict_salary() on all rows here, but we'll use mean salary for simplicity
    
    # Calculate Mean Salary and Variance (Risk) for the matched jobs
    recommendations = top_matches.groupby('Job_Title').agg(
        Mean_Salary=('Salary_USD', 'mean'),
        Salary_STD=('Salary_USD', 'std'),
        Count=('Salary_USD', 'count')
    ).reset_index()
    
    # Calculate a simple "Risk Score" (STD) and sort by highest Mean Salary
    recommendations['Salary_Risk'] = recommendations['Salary_STD'].fillna(0).apply(lambda x: 'High' if x > 20000 else 'Low')
    recommendations = recommendations.sort_values(by='Mean_Salary', ascending=False)
    
    return recommendations.head(5) # Top 5 recommendations

# --- 3. Streamlit Application UI ---

st.title("🚀 AI Career Path & Salary Optimization Recommender")
st.markdown("Enter your preferences to get personalized, high-value career recommendations based on market data and predictive modeling.")

# --- User Input Section ---
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

# --- Recommendation Button ---
if st.button("Generate My Optimal Career Recommendations"):
    if not selected_skills:
        st.warning("Please select at least one skill to receive recommendations.")
    else:
        st.header("2. Personalized Recommendations")
        
        # Get the top recommendations based on the user's inputs
        recommendations_df = get_recommendations(selected_skills, selected_risk)
        
        if recommendations_df.empty:
            st.warning(f"No strong matches found for your selected skills and the '{selected_risk}' risk level.")
        else:
            st.success("Here are the top career paths matching your profile:")
            
            # Display Recommendations
            st.dataframe(
                recommendations_df[['Job_Title', 'Mean_Salary', 'Salary_Risk']],
                column_config={
                    "Mean_Salary": st.column_config.NumberColumn("Predicted Mean Salary (USD)", format="$%d"),
                    "Job_Title": "Job Title",
                    "Salary_Risk": "Salary Volatility (Risk)"
                },
                hide_index=True
            )
            
            # Optional: Add a conclusion based on the highest-ranking job
            best_job = recommendations_df.iloc[0]
            st.markdown(f"**Top Recommendation:** The **{best_job['Job_Title']}** path offers the best blend of high earnings and alignment with your **{selected_risk}** risk preference, with an average salary of **${best_job['Mean_Salary']:.0f}**.")
