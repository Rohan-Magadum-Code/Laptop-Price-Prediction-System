from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Laptop Price Predictor API")

# Load once at startup
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

class LaptopFeatures(BaseModel):
    Company: str = "Dell"
    TypeName: str = "Notebook"
    Ram_GB: int = 8
    Weight: float = 2.0
    Touchscreen: int = 0
    IPS: int = 1
    ppi: float = 141.0
    Cpu_Name: str = "Intel"
    Cpu_Speed_GHz: float = 2.5
    Cpu_brand: str = "Intel Core i5"
    SSD: int = 256
    Gpu_Brand: str = "Intel"
    OS: str = "Windows"

@app.get("/")
def home():
    return {"message": "Laptop Price Predictor API is running"}

@app.post("/predict")
def predict(data: LaptopFeatures):
    try:
        input_df = pd.DataFrame([{
            "Company": data.Company,
            "TypeName": data.TypeName,
            "Ram (GB)": data.Ram_GB,
            "Weight": data.Weight,
            "Touchscreen": data.Touchscreen,
            "IPS": data.IPS,
            "ppi": data.ppi,
            "Cpu Name": data.Cpu_Name,
            "Cpu Speed (GHz)": data.Cpu_Speed_GHz,
            "Cpu brand": data.Cpu_brand,
            "SSD": data.SSD,
            "Gpu Brand": data.Gpu_Brand,
            "OS": data.OS
        }])

        processed_data = preprocessor.transform(input_df)

        prediction = model.predict(processed_data)[0]

        return {
            "predicted_price": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}