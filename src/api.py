from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
import util as util
import data_pipeline as data_pipeline
import preprocessing as preprocessing

config_data = util.load_config()
model_data = util.pickle_load(config_data["production_model_path"])


class ApiData(BaseModel):
    class Config:
        allow_population_by_field_name = True

    INTD_JK: float = Field(..., alias="INTD.JK")
    ULTJ_JK: float = Field(..., alias="ULTJ.JK")
    PDES_JK: float = Field(..., alias="PDES.JK")
    KICI_JK: float = Field(..., alias="KICI.JK")
    PGJO_JK: float = Field(..., alias="PGJO.JK")
    IKBI_JK: float = Field(..., alias="IKBI.JK")
    APII_JK: float = Field(..., alias="APII.JK")
    TLKM_JK: float = Field(..., alias="TLKM.JK")
    JKON_JK: float = Field(..., alias="JKON.JK")

data = {
    "INTD.JK": 1.0,
    "ULTJ.JK": 2.0,
    "PDES.JK": 3.0,
    "KICI.JK": 4.0,
    "PGJO.JK": 5.0,
    "IKBI.JK": 6.0,
    "APII.JK": 7.0,
    "TLKM.JK": 8.0,
    "JKON.JK": 9.0
}

api_data = ApiData(**data)


app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: ApiData):    
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
    uvicorn.run("api:app", host = "127.0.0.1", port = 8080)
