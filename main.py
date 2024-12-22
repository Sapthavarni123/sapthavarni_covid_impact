from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("final_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add fields for all features your model requires

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict/")
def predict(input_data: ModelInput):
    # Prepare input data for prediction
    input_array = np.array([[input_data.feature1, input_data.feature2, input_data.feature3]])
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}