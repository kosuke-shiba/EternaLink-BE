from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from location import update_location_data
from vector import update_vector_store
from llmchat import generate_response, get_family_data
from db import get_db

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    family_id: int
    user_id: int

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    result = generate_response(request.user_input, request.family_id, request.user_id, db)
    return result  # {"response": ..., "photo": ...}

@app.post("/update-location")
def update_location():
    message = update_location_data()
    return {"message": message}

@app.post("/update-vector")
def update_vector():
    message = update_vector_store()
    return {"message": message}

class FamilyRequest(BaseModel):
    family_id: int
    user_id: int

@app.post("/family-info")
def get_family_info(request: FamilyRequest, db: Session = Depends(get_db)):
    data = get_family_data(request.family_id, request.user_id, db)
    return {"family": data}
