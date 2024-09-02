from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle


app = FastAPI()

class ScoringItem(BaseModel):
    Logical_quotient_rating: int
    hackathons: int
    coding_skills_rating: int
    public_speaking_points: int
    self_learning_capability: int
    extra_courses_did: int
    certifications: int
    workshops: int
    reading_and_writing_skills: int
    memory_capability_score: int
    interested_subjects: int
    interested_career_area: int
    type_of_company_want_to_settle_in: int
    taken_inputs_from_seniors_or_elders: int
    interested_type_of_books: int
    A_management: int
    A_Technical:int
    B_smart_worker: int
    B_hard_worker:int
    worked_in_teams: int
    introvert: int


with open('models/weights.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/predict')
async def scoring_endpoint(data:ScoringItem):
    try:
        input_vars = [data.Logical_quotient_rating, data.hackathons, data.coding_skills_rating,
                      data.public_speaking_points, data.self_learning_capability, data.extra_courses_did,data.certifications,data.workshops,data.reading_and_writing_skills
                      ,data.interested_subjects,data.memory_capability_score,data.interested_career_area,data.type_of_company_want_to_settle_in
                      ,data.taken_inputs_from_seniors_or_elders,data.interested_type_of_books,data.A_management,data.A_Technical,data.B_hard_worker,data.B_smart_worker,data.worked_in_teams,data.introvert]
        yhat = model.predict([input_vars])
        return {"prediction":str(yhat)}
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

