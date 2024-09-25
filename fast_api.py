from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the models
log_reg_model = joblib.load('logistic_regression_model.joblib')
xgb_model = joblib.load('xgboost_model.joblib')

# Initialize FastAPI app
app = FastAPI()



@app.post("/predict/logistic_reg")
def predict_logistic_reg(data: InputData):
    # Convert input data to DataFrame for the model
    input_df = pd.DataFrame([data.dict()])
    
    # Perform any preprocessing here if needed (e.g., scaling)
    
    # Make prediction
    prediction = log_reg_model.predict(input_df)
    return {"churn_prediction": int(prediction[0])}



@app.post("/predict/xgboost")
def predict_xgboost(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = xgb_model.predict(input_df)
    return {"churn_prediction": int(prediction[0])}

# To run the app, use the command: uvicorn your_script_name:app --reload
import nest_asyncio
from pyngrok import ngrok

nest_asyncio.apply()

public_url = ngrok.connect(8000)
print("Public URL:", public_url)

