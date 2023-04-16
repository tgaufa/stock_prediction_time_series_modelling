from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import data_pipeline as data_pipeline
import preprocessing as preprocessing

config_data = util.load_config()
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    ticker1 : float
    ticker2 : float
    ticker3 : float
    ticker4 : float
    ticker5 : float
    ticker6 : float
    ticker7 : float
    ticker8 : float
    ticker9 : float
    ticker10 : float

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

    # Check range data
    try:
        data_pipeline.check_data(data, config_data, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    # Inverse tranform
    #y_pred = list(le_encoder.inverse_transform(y_pred))[0] 

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
