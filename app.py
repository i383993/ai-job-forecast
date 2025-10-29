import streamlit as st
import pandas as pd
import plotly.express as px
import joblib 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split # New Import for Training
from sklearn.ensemble import RandomForestRegressor # New Model Import

# --- Configuration and Data Loading ---

# Assuming 'ai_job_market_insights.csv' is available. 
# It is assumed to have columns: 
# 'Job_Title', 'Salary_USD', 'Industry', 'Automation_Risk', 'AI_Adoption_Level'
try:
    df = pd.read_csv("ai_job_market_insights.csv")
    
    # Simple Data Pre-processing for robust visualization
    if 'Salary_USD' in df.columns:
        df['Salary_USD'] = pd.to_numeric(df['Salary_USD'], errors='coerce')
    
    # Map Automation Risk to readable strings if stored as numbers (0, 1, 2)
    risk_map = {0: 'Low/None', 1: 'Moderate', 2: 'High/Advanced'}
    if df['Automation_Risk'].dtype in ['int64', 'float64']:
        df['Automation_Risk_Label'] = df['Automation_Risk'].map(risk_map).fillna('Unknown')
    else:
        df['Automation_Risk_Label'] = df['Automation_Risk']
        # Reverse map for consistency, assuming the original training script converted this column to int
        df['Automation_Risk'] = df['Automation_Risk_Label'].map({v: k for k, v in risk_map.items()}).fillna(-1) 

    # Ensure all text is stripped of whitespace for consistent encoding
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Global store for encoders (CRITICAL for consistent prediction)
    ENCODER_MAP = {}
    
    # Replicate the LabelEncoding step from the training script
    # We fit the encoders on the full dataset so the model can consistently predict.
    def fit_encoders(data_frame):
        # Identify original categorical columns used as features
        categorical_cols = data_frame.select_dtypes(include='object').columns.tolist()
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on all unique values in the column
            le.fit(data_frame[col].astype(str).unique())
            ENCODER_MAP[col] = le
            # Transform the columns in the main DataFrame
            data_frame[col] = le.transform(data_frame[col].astype(str))
    
    # Run the setup - This transforms the categorical columns to their encoded integer values
    fit_encoders(df)

    # --- Robust Feature Handling for Prediction (The Fix) ---
    
    # 1. Get the list of ALL features the model was trained on (all numeric columns minus the target)
    ALL_FEATURE_COLS = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Salary_USD' in ALL_FEATURE_COLS:
        ALL_FEATURE_COLS.remove('Salary_USD')

    # 2. Identify the columns that are based on user input (the original object columns)
    # The columns that the model expects are the 4 user-selected plus the 5 imputed
    USER_INPUT_COLS = ['AI_Adoption_Level', 'Automation_Risk_Label', 'Industry', 'Job_Title']
    
    # 3. Identify columns that need imputation (The "other 5" features)
    IMPUTATION_COLS = [col for col in ALL_FEATURE_COLS if col not in USER_INPUT_COLS]
    
    # 4. Global store for imputed feature means
    NUMERIC_MEANS = {}
    for col in IMPUTATION_COLS:
        NUMERIC_MEANS[col] = df[col].mean()

    # --- New Model Training and Loading ---
    
    def train_random_forest(data_frame, feature_cols, target_col='Salary_USD'):
        # 1. Prepare data (use ALL_FEATURE_COLS which has 9 features)
        X = data_frame[feature_cols].copy()
        y = data_frame[target_col].copy()

        # Drop rows with NaN in target or features (critical for training)
        valid_indices = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        if X.empty:
            st.warning("Cannot train model: Data is empty after handling NaNs. Prediction will not work.")
            return None

        # 2. Train the model
        # Using a small max_depth for faster training in a web app environment
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=8)
        rf_model.fit(X, y)
        
        return rf_model

    # Attempt to load or train a model (Replacing the old loading logic)
    regression_model = None
    model_loaded = False
    model_name = "None"
    
    try:
        # Try to train the new Random Forest model immediately
        regression_model = train_random_forest(df, ALL_FEATURE_COLS)
        if regression_model is not None:
             model_loaded = True
             model_name = "Random Forest Regressor (Live Trained)"
             st.sidebar.success(f"Model Ready: {model_name}")
        else:
            # If training failed, try to load the original pre-trained model as a fallback
            regression_model = joblib.load("multiple_regression_model.pkl")
            model_loaded = True
            model_name = "Linear Regression (Fallback)"
            st.sidebar.warning(f"Using Fallback Model: {model_name}")

    except FileNotFoundError:
        st.error("Error: Could not load any prediction model file (multiple_regression_model.pkl).")
        model_loaded = False
    except Exception as e:
        st.error(f"Error during model setup: {e}")
        model_loaded = False


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
st.markdown("Explore macro market trends, dive deep into job risks, or predict potential salary based on job characteristics.")

if data_loaded:
    # --- Main Navigation ---
    analysis_mode = st.radio(
        "Select Analysis Mode",
        ('Industry Analysis', 'Job Title Deep Dive', 'Salary Prediction'), # Added new mode
        horizontal=True,
        index=0
    )

    # --- Section 1: Industry Analysis ---
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

            # Visualization 2: Automation Risk Distribution for the Selected Industry
            fig_auto = px.histogram(
                filtered_df, 
                x="Automation_Risk_Label", 
                color="AI_Adoption_Level",
                title=f"Automation Risk Distribution in {selected_industry}",
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
            # Note: Displaying the original 'Automation_Risk' which is now an integer, 
            # as the original training likely used the integer version of the column.
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
                    marginal="box", 
                    title=f"Salary Distribution for: {selected_job}",
                    labels={'Salary_USD': 'Annual Salary (USD)'},
                    color_discrete_sequence=['#4CAF50']
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

    # --- Section 3: Salary Prediction ---
    elif analysis_mode == 'Salary Prediction':
        st.header("3. Salary Prediction Model")
        st.markdown("Estimate a job's potential salary based on its core characteristics.")

        if not model_loaded:
            st.warning("Prediction is unavailable because the model could not be loaded or trained.")
        else:
            # 1. Collect user inputs
            # Use the original object columns for user selection, but the encoded columns for prediction
            job_titles = sorted(df["Job_Title"].unique())
            industries = sorted(df["Industry"].unique())
            # For risk levels, use the readable labels
            risk_labels = sorted(df["Automation_Risk_Label"].unique())
            ai_adoption_levels = sorted(df["AI_Adoption_Level"].unique())
            
            with st.form("salary_prediction_form"):
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    input_job = st.selectbox("Job Title", job_titles, key="pred_job")
                    input_risk = st.selectbox("Automation Risk Level", risk_labels, key="pred_risk")
                
                with col_b:
                    input_industry = st.selectbox("Industry", industries, key="pred_industry")
                    input_ai_adoption = st.selectbox("Industry AI Adoption Level", ai_adoption_levels, key="pred_ai_adoption")
                
                predict_button = st.form_submit_button("Predict Salary")

            if predict_button:
                try:
                    # 2. Construct the 9-feature vector for prediction
                    
                    # 2.1 Start with a dictionary of all features, initialized with imputed means
                    prediction_features = {}
                    for col, mean_val in NUMERIC_MEANS.items():
                        prediction_features[col] = mean_val
                    
                    # 2.2 Overlay the 4 user-provided features (Encoded values)
                    # We use the ENCODER_MAP to transform the user's string input into the trained integer value.
                    
                    # Job Title is the original categorical name
                    prediction_features['Job_Title'] = ENCODER_MAP['Job_Title'].transform([input_job])[0]
                    
                    # Industry is the original categorical name
                    prediction_features['Industry'] = ENCODER_MAP['Industry'].transform([input_industry])[0]
                    
                    # Automation_Risk_Label is the original categorical name for the risk label
                    prediction_features['Automation_Risk_Label'] = ENCODER_MAP['Automation_Risk_Label'].transform([input_risk])[0]
                    
                    # AI_Adoption_Level is the original categorical name
                    prediction_features['AI_Adoption_Level'] = ENCODER_MAP['AI_Adoption_Level'].transform([input_ai_adoption])[0]

                    # 2.3 Create the final input array, ensuring the column order matches ALL_FEATURE_COLS
                    X_predict_ordered = [prediction_features[col] for col in ALL_FEATURE_COLS]
                    
                    # 3. Make Prediction (requires a 2D array [[]])
                    X_predict_array = np.array([X_predict_ordered])
                    
                    prediction = regression_model.predict(X_predict_array)[0]
                    predicted_salary = max(0, prediction) # Salary cannot be negative
                    
                    # 4. Display Result
                    st.subheader("💰 Predicted Annual Salary")
                    
                    # Display the final prediction
                    st.success(f"The predicted salary for a **{input_job}** in the **{input_industry}** industry is approximately:")
                    st.success(f"## ${predicted_salary:,.0f}")
                    
                    # Add interpretation based on the risk level
                    if input_risk == 'High/Advanced':
                         st.warning(f"Note: This role has a **High Automation Risk**. While the salary is good, upskilling is crucial for long-term career stability.")
                    elif input_risk == 'Low/None':
                         st.info("This role has a **Low Automation Risk**, suggesting a high level of human judgment and stability in the face of current AI tools.")

                except KeyError as e:
                    st.error(f"Prediction Error: A feature value ({e}) was not present in the original training data. Please try another selection.")
                except Exception as e:
                    # Display the feature array shape for debugging if an unexpected error occurs
                    st.error(f"An unexpected error occurred during prediction: {e}")
                    # This line is crucial for diagnosis and will show the user what went wrong with the shape
                    st.error(f"Debug Info: Predicted feature count: {len(X_predict_ordered)}. Expected feature count: 9.")

st.caption(f"Developed by Isaiah Panicker • Data Science Project • Using {model_name}")
