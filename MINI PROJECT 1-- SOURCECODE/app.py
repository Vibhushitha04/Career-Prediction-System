from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle


app = FastAPI()

class ScoringItem(BaseModel):
    Age: int
    SystolicBP: int
    DiastolicBP: int
    BS: int
    BodyTemp: int
    HeartRate: int


with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/predict')
async def scoring_endpoint(data:ScoringItem):
    try:
        input_vars = [data.Age, data.SystolicBP, data.DiastolicBP,
                      data.BS, data.BodyTemp, data.HeartRate]
        yhat = model.predict([input_vars])
        return {"prediction":int(yhat)}
    except Exception as e:
        return {"error": str(e)}, 422
    
@app.options('/predict')
async def options_endpoint(response: Response):
    response.headers["Allow"] = "POST, OPTIONS"
    return {}

# Add a middleware to allow CORS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

