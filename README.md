# longevity_analytics_platform_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
from typing import Optional
import json
warnings.filterwarnings('ignore')

# Configure Streamlit page
def configure_page():
    try:
        st.set_page_config(
            page_title="Longevity Analytics Platform",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

# Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            border-left: 4px solid #667eea;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

# Safe data type converter
def safe_convert_dtypes(df):
    """Convert object columns to appropriate data types safely"""
    df = df.copy()
    for col in df.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Try to convert to numeric
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
            
        # If still object and has few unique values, try categorical
        if df[col].dtype == 'object' and df[col].nunique() < 50:
            try:
                df[col] = df[col].astype('category')
            except Exception:
                pass
                
    return df

# Sample data generator
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'participant_id': range(1, n_samples + 1),
        'age': np.random.normal(65, 15, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'bmi': np.random.normal(26, 4, n_samples),
        'blood_pressure_systolic': np.random.normal(130, 20, n_samples),
        'blood_pressure_diastolic': np.random.normal(80, 12, n_samples),
        'cholesterol_ldl': np.random.normal(110, 30, n_samples),
        'cholesterol_hdl': np.random.normal(55, 15, n_samples),
        'glucose_fasting': np.random.normal(95, 20, n_samples),
        'crp_level': np.random.exponential(2, n_samples),
        'telomere_length': np.random.normal(1.0, 0.2, n_samples),
        'epigenetic_age': np.random.normal(65, 10, n_samples),
        'vo2_max': np.random.normal(30, 8, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'exercise_hours_week': np.random.exponential(3, n_samples),
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.5, 0.3, 0.2]),
        'alcohol_consumption': np.random.exponential(2, n_samples),
        'intervention_type': np.random.choice(['None', 'Caloric Restriction', 'Exercise', 'Metformin', 'Rapamycin'], 
                                             n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        'intervention_duration_months': np.random.exponential(12, n_samples),
        'healthspan_years': np.random.normal(75, 10, n_samples)
    }
    df = pd.DataFrame(data)
    df['age'] = np.clip(df['age'], 30, 100)
    df['bmi'] = np.clip(df['bmi'], 15, 45)
    df['blood_pressure_systolic'] = np.clip(df['blood_pressure_systolic'], 80, 200)
    df['blood_pressure_diastolic'] = np.clip(df['blood_pressure_diastolic'], 50, 120)
    df['healthspan_years'] = np.clip(df['healthspan_years'], df['age'], 100)
    df['healthspan_years'] += (df['telomere_length'] * 5) - (df['crp_level'] * 2) + (df['vo2_max'] * 0.3)
    df['epigenetic_age'] = df['age'] + np.random.normal(0, 5, n_samples)
    return df

# Safe loader
def safe_load_data(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = safe_convert_dtypes(df)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Data Collection Page
def data_collection_page():
    st.header("📥 Data Collection & Import")
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file with longevity data", type=['csv'])
        
        if uploaded_file is not None:
            df = safe_load_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                st.subheader("Dataset Information")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Records", df.shape[0])
                with c2:
                    st.metric("Total Features", df.shape[1])
                with c3:
                    st.metric("Missing Values", int(df.isnull().sum().sum()))
                
                st.subheader("Data Types")
                dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                st.dataframe(dtype_df)
    
    with col2:
        st.subheader("Or Use Sample Data")
        st.write("Generate a realistic sample dataset for demonstration:")
        sample_size = st.slider("Sample Size", 100, 5000, 1000)
        
        if st.button("🎲 Generate Sample Dataset"):
            with st.spinner("Generating sample data..."):
                sample_df = generate_sample_data(sample_size)
                st.session_state.data = sample_df
                st.success(f"✅ Sample dataset generated with {sample_size} records!")
                time.sleep(0.5)
                st.rerun()

        if st.session_state.data is not None:
            st.subheader("Quick Stats")
            df = st.session_state.data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            st.write(f"**Numeric Columns:** {len(numeric_cols)}")
            st.write(f"**Categorical Columns:** {len(categorical_cols)}")
            
            if 'healthspan_years' in df.columns:
                st.metric("Avg Healthspan", f"{df['healthspan_years'].mean():.1f} years")

# Data Cleaning Page
def data_cleaning_page():
    st.header("🧹 Data Cleaning & Preprocessing")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
        
    df = st.session_state.data.copy()
    
    st.subheader("Data Quality Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values by Column:**")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            # Create a simple bar chart without complex data types
            fig = px.bar(
                x=missing_data.index.astype(str), 
                y=missing_data.values, 
                title="Missing Values Count",
                labels={'x': 'Columns', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    with col2:
        st.write("**Data Types Summary:**")
        type_counts = df.dtypes.astype(str).value_counts()
        fig = px.pie(
            values=type_counts.values, 
            names=type_counts.index, 
            title="Data Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Cleaning Options")
    remove_outliers = st.checkbox("Remove Outliers", value=True)
    fill_missing = st.checkbox("Fill Missing Values", value=True)
    normalize_data = st.checkbox("Normalize Data", value=False)
    remove_duplicates = st.checkbox("Remove Duplicates", value=True)
    
    fill_method = None
    outlier_method = None
    
    if fill_missing:
        fill_method = st.selectbox("Fill Method", ["Mean", "Median", "Mode", "Forward Fill"])    
    if remove_outliers:
        outlier_method = st.selectbox("Outlier Detection Method", ["Z-Score", "IQR"])    
    
    if st.button("🧹 Apply Cleaning"):
        with st.spinner("Cleaning data..."):
            cleaned_df = df.copy()
            original_shape = cleaned_df.shape
            
            # Remove duplicates
            if remove_duplicates:
                cleaned_df = cleaned_df.drop_duplicates()
            
            # Fill missing values
            if fill_missing and cleaned_df.isnull().sum().sum() > 0:
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
                
                if fill_method == "Mean":
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
                elif fill_method == "Median":
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
                elif fill_method == "Mode":
                    for col in cleaned_df.columns:
                        if col in numeric_cols:
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 0)
                        else:
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else "Unknown")
                elif fill_method == "Forward Fill":
                    cleaned_df = cleaned_df.fillna(method='ffill')
            
            # Remove outliers
            if remove_outliers:
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if outlier_method == "Z-Score":
                        z_scores = np.abs(stats.zscore(cleaned_df[col].dropna()))
                        mask = z_scores < 3
                        valid_indices = cleaned_df[col].dropna().index[mask]
                        cleaned_df = cleaned_df.loc[valid_indices]
                    elif outlier_method == "IQR":
                        Q1 = cleaned_df[col].quantile(0.25)
                        Q3 = cleaned_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
            
            # Normalize data
            if normalize_data:
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                scaler = StandardScaler()
                cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])
            
            st.session_state.cleaned_data = cleaned_df
            st.success("✅ Cleaning completed!")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Original Records", original_shape[0])
            with c2:
                st.metric("Cleaned Records", cleaned_df.shape[0])
            with c3:
                reduction = ((original_shape[0] - cleaned_df.shape[0]) / original_shape[0]) * 100
                st.metric("Records Removed", f"{reduction:.1f}%")
            
            st.subheader("Cleaned Data Preview")
            st.dataframe(cleaned_df.head())

# Exploratory Analysis Page
def exploratory_analysis_page():
    st.header("📈 Exploratory Data Analysis")
    
    data_source = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
    if data_source is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
        
    df = data_source.copy()
    
    st.subheader("📊 Summary Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Variables Statistics:**")
            st.dataframe(df[numeric_cols].describe().round(2))
        
        with col2:
            st.write("**Distribution Analysis:**")
            selected_var = st.selectbox("Select variable to analyze", numeric_cols)
            
            if selected_var in df.columns:
                fig = make_subplots(
                    rows=2, cols=1, 
                    subplot_titles=('Histogram', 'Box Plot'), 
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Histogram(x=df[selected_var].dropna(), name='Distribution'), 
                    row=1, col=1
                )
                fig.add_trace(
                    go.Box(y=df[selected_var].dropna(), name='Box Plot'), 
                    row=2, col=1
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        st.subheader("📋 Categorical Variables Analysis")
        selected_cat = st.selectbox("Select categorical variable", categorical_cols)
        
        c1, c2 = st.columns(2)
        with c1:
            cat_counts = df[selected_cat].value_counts()
            fig = px.bar(
                x=cat_counts.index.astype(str), 
                y=cat_counts.values, 
                title=f"Distribution of {selected_cat}",
                labels={'x': selected_cat, 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            if 'healthspan_years' in df.columns:
                fig = px.box(
                    df, 
                    x=selected_cat, 
                    y='healthspan_years', 
                    title=f"Healthspan by {selected_cat}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    if 'age' in df.columns and 'healthspan_years' in df.columns:
        st.subheader("📈 Age vs Healthspan Analysis")
        
        # Create age groups
        age_bins = [0, 40, 50, 60, 70, 80, 100]
        age_labels = ['<40', '40-50', '50-60', '60-70', '70-80', '>80']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        
        fig = px.scatter(
            df, 
            x='age', 
            y='healthspan_years', 
            color='age_group', 
            title="Age vs Healthspan Relationship", 
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        age_healthspan = df.groupby('age_group')['healthspan_years'].mean().reset_index()
        fig = px.bar(
            age_healthspan, 
            x='age_group', 
            y='healthspan_years', 
            title="Average Healthspan by Age Group"
        )
        st.plotly_chart(fig, use_container_width=True)

# Correlation Analysis Page
def correlation_analysis_page():
    st.header("🔗 Correlation & Relationship Analysis")
    
    data_source = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
    if data_source is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
        
    df = data_source.copy()
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        st.subheader("🔢 Correlation Matrix")
        corr_method = st.selectbox("Correlation Method", ["Pearson", "Spearman", "Kendall"])
        
        try:
            corr_matrix = numeric_df.corr(method=corr_method.lower())
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale="RdBu_r", 
                title=f"Correlation Matrix ({corr_method})"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🎯 Strongest Correlations")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i], 
                        'Variable 2': corr_matrix.columns[j], 
                        'Correlation': corr_matrix.iloc[i,j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df['Absolute Correlation'] = corr_df['Correlation'].abs()
            corr_df = corr_df.sort_values('Absolute Correlation', ascending=False)
            
            top_n = st.slider("Number of top correlations to show", 5, 20, 10)
            st.dataframe(corr_df.head(top_n).round(3))
            
            st.subheader("🔍 Detailed Correlation Analysis")
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Select first variable", numeric_df.columns)
            with col2:
                var2 = st.selectbox("Select second variable", numeric_df.columns, index=min(1, len(numeric_df.columns)-1))
            
            if var1 != var2:
                try:
                    valid_data = df[[var1, var2]].dropna()
                    if len(valid_data) > 1:
                        corr_coef, p_value = pearsonr(valid_data[var1], valid_data[var2])
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
                        with c2:
                            st.metric("P-value", f"{p_value:.4f}")
                        with c3:
                            significance = "Significant" if p_value < 0.05 else "Not Significant"
                            st.metric("Significance", significance)
                        
                        fig = px.scatter(
                            df, 
                            x=var1, 
                            y=var2, 
                            title=f"{var1} vs {var2}", 
                            trendline="ols"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data points for correlation analysis")
                except Exception as e:
                    st.error(f"Error calculating correlation: {str(e)}")
        except Exception as e:
            st.error(f"Error creating correlation matrix: {str(e)}")
    else:
        st.warning("⚠️ Need at least 2 numeric columns for correlation analysis!")

# Predictive Modeling Page
def predictive_modeling_page():
    st.header("🤖 Predictive Modeling for Healthspan")
    
    data_source = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
    if data_source is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
        
    df = data_source.copy()
    
    if 'healthspan_years' not in df.columns:
        st.warning("⚠️ Dataset must contain 'healthspan_years' column for prediction!")
        return
        
    st.subheader("🎯 Model Configuration")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'healthspan_years' in numeric_cols:
        numeric_cols.remove('healthspan_years')
        
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    features = numeric_cols.copy()
    
    # Encode categorical variables
    if categorical_cols:
        for col in categorical_cols:
            try:
                encoded_col = col + '_encoded'
                le = LabelEncoder()
                df[encoded_col] = le.fit_transform(df[col].astype(str))
                features.append(encoded_col)
            except Exception:
                st.warning(f"Could not encode column: {col}")
    
    selected_features = st.multiselect("Choose features", features, default=features[:min(5, len(features))])
    
    if not selected_features:
        st.info("Select at least one feature to train the model")
        return
        
    # Prepare data
    X = df[selected_features]
    y = df['healthspan_years']
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        st.error("No valid data available for training after removing missing values")
        return
        
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    except Exception as e:
        st.error(f"Error splitting data: {str(e)}")
        return
        
    st.subheader("🤖 Model Selection & Training")
    model_type = st.selectbox("Choose Model", ["Random Forest", "Gradient Boosting", "Linear Regression", "Elastic Net"])    
    
    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            try:
                if model_type == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Elastic Net":
                    model = ElasticNet(random_state=42)
                
                model.fit(X_train, y_train)
                st.session_state.model = model
                
                y_pred = model.predict(X_test)
                st.session_state.predictions = y_pred
                
                st.success("✅ Model trained successfully!")
                
                st.subheader("📊 Model Performance")
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    mse = mean_squared_error(y_test, y_pred)
                    st.metric("MSE", f"{mse:.2f}")
                with c2:
                    rmse = np.sqrt(mse)
                    st.metric("RMSE", f"{rmse:.2f}")
                with c3:
                    mae = mean_absolute_error(y_test, y_pred)
                    st.metric("MAE", f"{mae:.2f}")
                with c4:
                    r2 = r2_score(y_test, y_pred)
                    st.metric("R² Score", f"{r2:.3f}")
                
                st.subheader("📈 Predictions vs Actual")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, 
                    y=y_pred, 
                    mode='markers', 
                    name='Predictions', 
                    marker=dict(opacity=0.6)
                ))
                
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], 
                    y=[min_val, max_val], 
                    mode='lines', 
                    name='Perfect Prediction', 
                    line=dict(dash='dash')
                ))
                
                fig.update_layout(
                    title='Predicted vs Actual Healthspan', 
                    xaxis_title='Actual Healthspan (years)', 
                    yaxis_title='Predicted Healthspan (years)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if hasattr(model, 'feature_importances_'):
                    st.subheader("🎯 Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': selected_features, 
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df.head(10), 
                        x='Importance', 
                        y='Feature', 
                        orientation='h', 
                        title='Top 10 Feature Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(importance_df.round(3))
                    
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    if st.session_state.model is not None:
        st.subheader("🔮 Make New Predictions")
        st.write("Enter values for prediction:")
        
        input_data = {}
        for feature in selected_features:
            if feature in df.select_dtypes(include=[np.number]).columns:
                default_val = float(X[feature].mean()) if len(X) > 0 else 0.0
                input_data[feature] = st.number_input(
                    f"{feature}", 
                    value=default_val, 
                    step=0.1
                )
            else:
                unique_values = sorted(X[feature].unique()) if len(X) > 0 else [0]
                input_data[feature] = st.selectbox(
                    f"{feature}", 
                    unique_values, 
                    index=min(len(unique_values)//2, len(unique_values)-1)
                )
        
        if st.button("🎯 Predict Healthspan"):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = st.session_state.model.predict(input_df)[0]
                st.success(f"🎯 Predicted Healthspan: {prediction:.1f} years")
                
                std_error = np.std(st.session_state.predictions) if st.session_state.predictions is not None else 5
                ci_lower = prediction - 1.96 * std_error
                ci_upper = prediction + 1.96 * std_error
                st.write(f"95% Confidence Interval: [{ci_lower:.1f}, {ci_upper:.1f}] years")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Reports Page
def reports_page():
    st.header("📋 Analysis Reports & Insights")
    
    data_source = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
    if data_source is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
        
    df = data_source.copy()
    
    st.subheader("📊 Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        if 'healthspan_years' in df.columns:
            avg_healthspan = df['healthspan_years'].mean()
            st.metric("Avg Healthspan", f"{avg_healthspan:.1f} years")
    
    with c2:
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            st.metric("Avg Age", f"{avg_age:.1f} years")
    
    with c3:
        if 'bmi' in df.columns:
            avg_bmi = df['bmi'].mean()
            st.metric("Avg BMI", f"{avg_bmi:.1f}")
    
    with c4:
        if 'intervention_type' in df.columns and 'healthspan_years' in df.columns:
            intervention_stats = df.groupby('intervention_type')['healthspan_years'].mean()
            if not intervention_stats.empty:
                intervention_effectiveness = intervention_stats.max()
                best_intervention = intervention_stats.idxmax()
                st.metric("Best Intervention", f"{best_intervention}: {intervention_effectiveness:.1f} years")
    
    st.subheader("💡 Key Insights")
    insights = []
    
    if 'healthspan_years' in df.columns and 'age' in df.columns:
        healthspan_gap = df['healthspan_years'].mean() - df['age'].mean()
        if healthspan_gap > 0:
            insights.append(f"✅ Average healthspan exceeds current age by {healthspan_gap:.1f} years")
        else:
            insights.append(f"⚠️ Average healthspan is {abs(healthspan_gap):.1f} years below current age")
    
    if 'bmi' in df.columns:
        overweight_pct = (df['bmi'] > 25).mean() * 100
        insights.append(f"📊 {overweight_pct:.1f}% of participants have BMI > 25")
    
    if 'intervention_type' in df.columns and 'healthspan_years' in df.columns:
        intervention_stats = df.groupby('intervention_type')['healthspan_years'].mean()
        if not intervention_stats.empty:
            best_intervention = intervention_stats.idxmax()
            best_healthspan = intervention_stats.max()
            insights.append(f"🏆 {best_intervention} shows highest average healthspan: {best_healthspan:.1f} years")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'healthspan_years' in numeric_cols and len(numeric_cols) > 1:
        correlations = []
        for col in numeric_cols:
            if col != 'healthspan_years':
                try:
                    valid_data = df[[col, 'healthspan_years']].dropna()
                    if len(valid_data) > 1:
                        corr, _ = pearsonr(valid_data[col], valid_data['healthspan_years'])
                        correlations.append((col, abs(corr)))
                except Exception:
                    pass
        
        if correlations:
            correlations.sort(key=lambda x: x[1], reverse=True)
            best_biomarker = correlations[0][0]
            insights.append(f"🔬 {best_biomarker} shows strongest correlation with healthspan")
    
    for insight in insights:
        st.write(f"• {insight}")
    
    st.subheader("📉 Visual Insights")
    
    if 'healthspan_years' in df.columns:
        fig = px.histogram(df, x='healthspan_years', nbins=30, title='Healthspan Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    if 'intervention_type' in df.columns and 'healthspan_years' in df.columns:
        fig = px.box(df, x='intervention_type', y='healthspan_years', title='Healthspan by Intervention')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📥 Download Report")
    if st.button("📊 Generate Full Report"):
        report = []
        report.append("LONGEVITY ANALYTICS PLATFORM - ANALYSIS REPORT")
        report.append("Generated on: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        report.append("")
        report.append("DATASET SUMMARY")
        report.append(f"Total Records: {len(df)}")
        report.append(f"Total Features: {len(df.columns)}")
        report.append("")
        
        if 'healthspan_years' in df.columns:
            report.append("HEALTHSPAN ANALYSIS")
            report.append(f"Mean Healthspan: {df['healthspan_years'].mean():.2f} years")
            report.append(f"Median Healthspan: {df['healthspan_years'].median():.2f} years")
            report.append(f"Std Dev: {df['healthspan_years'].std():.2f} years")
            report.append("")
        
        report.append("KEY INSIGHTS")
        for insight in insights:
            report.append(f"• {insight}")
        
        report_text = "\n".join(report)
        
        st.download_button(
            label="📥 Download Report (TXT)", 
            data=report_text, 
            file_name=f"longevity_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt", 
            mime="text/plain"
        )
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data (CSV)", 
            data=csv, 
            file_name=f"processed_longevity_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            mime="text/csv"
        )

# Main function
def main():
    configure_page()
    apply_custom_css()
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🧬 Longevity Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('### Advanced Healthspan Intelligence & Biomarker Analysis Platform')
    
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Select a section:", [
        "📥 Data Collection",
        "🧹 Data Cleaning", 
        "📈 Exploratory Analysis",
        "🔗 Correlation Analysis",
        "🤖 Predictive Modeling",
        "📋 Reports"
    ])
    
    if "script_run_context_warning" not in st.session_state:
        st.session_state.script_run_context_warning = True
        st.sidebar.markdown("""
        <div class="warning-box">
            <strong>Note:</strong> If you see a ScriptRunContext warning, it can be safely ignored when running in bare mode. Run with `streamlit run` for full context.
        </div>
        """, unsafe_allow_html=True)
    
    if page == "📥 Data Collection":
        data_collection_page()
    elif page == "🧹 Data Cleaning":
        data_cleaning_page()
    elif page == "📈 Exploratory Analysis":
        exploratory_analysis_page()
    elif page == "🔗 Correlation Analysis":
        correlation_analysis_page()
    elif page == "🤖 Predictive Modeling":
        predictive_modeling_page()
    elif page == "📋 Reports":
        reports_page()
    
    st.markdown('---')
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "<p>🧬 Longevity Analytics Platform | Advanced Healthspan Intelligence</p>"
        "<p>Built with Python, Streamlit, and Machine Learning</p>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

#streamlit run "d:/My Data/ALL CODE PROJECTS/longevity_analytics_platform.py" use your file folder name to run it also save it
