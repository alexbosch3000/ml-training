from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import numpy as np
import json

# load model
clf = load('lr.joblib')

# Define a Data Model for your feature Matrix
class WineProperties(BaseModel):
    alcohol: float
    volatile_acidity: float
    # This data model is easy to scale to add more features

# Make a prediction using a Data Mode with Pydantic BaseModel
# Note that this method is agnostic to the number or features.
# However, you have to be sure that the model is trained with the right number of features
def get_prediction_args(**kwargs):
    
    # Define your features Matrix
    data = list(kwargs.values())
    X = np.array(data).reshape(1,-1)

    # Get the prediction and the predict probability
    # Note that numpy arrays cannot be send in to the final answer.
    y_pred = clf.predict(X)[0]  
    y_proba = clf.predict_proba(X)[0].tolist()  

    # Send the prediction value and probability in dictionary form. 
    return {'prediction': int(y_pred), 'probability': y_proba}


# initiate API
app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello stranger! This API allow you to evaluate the quality of red wine. Go to the /docs for more details."}


# define the predict endpoint
@app.post("/predict")
async def predict(WineProperties:WineProperties):

    # Get data from json format to dictionary
    WineProperties_dict = WineProperties.model_dump()

    # Get predictions
    pred = get_prediction_args(**WineProperties_dict)

    # Convert prediction to json reponse
    return json.dumps(pred)