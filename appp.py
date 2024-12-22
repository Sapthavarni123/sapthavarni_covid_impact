import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the model from MLflow
model_uri = "models:/model_name/1"  # Update model_name and version accordingly
model = mlflow.sklearn.load_model(model_uri)

# Define the preprocessing pipeline (same as during training)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('minmax', MinMaxScaler()),
    ('log_transform', FunctionTransformer(np.log1p, validate=True))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Increased_Work_Hours', 'Hours_Worked_Per_Day', 'Meetings_Per_Day', 'Commuting_Changes']),
        ('cat', categorical_transformer, ['Sector', 'Stress_Level'])
    ]
)

# Full pipeline with preprocessor and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Streamlit app UI
st.title("COVID Impact on Work Classification")
st.write("This app predicts productivity change based on employee work details and sector information.")

# Collect user input
sector = st.selectbox('Sector', ['Retail', 'Education', 'Healthcare'])
stress_level = st.selectbox('Stress Level', ['Low', 'Medium', 'High'])
increased_work_hours = st.number_input('Increased Work Hours (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
hours_worked_per_day = st.number_input('Hours Worked Per Day', min_value=0, max_value=24, step=1)
meetings_per_day = st.number_input('Meetings Per Day', min_value=0, max_value=10, step=1)
commuting_changes = st.number_input('Commuting Changes (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)

# Prediction button
if st.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'Sector': [sector],
        'Stress_Level': [stress_level],
        'Increased_Work_Hours': [increased_work_hours],
        'Hours_Worked_Per_Day': [hours_worked_per_day],
        'Meetings_Per_Day': [meetings_per_day],
        'Commuting_Changes': [commuting_changes]
    })
    
    # Predict the productivity change
    prediction = pipeline.predict(input_data)
    
    # Display the result
    st.write(f"Predicted Productivity Change: {prediction[0]}")
