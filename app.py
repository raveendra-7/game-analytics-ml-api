import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

model = joblib.load('game_rating_model.joblib')

app = FastAPI(title="Game Analytics ML API", version="1.0")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameFeatures(BaseModel):
    genre: str
    is_multiplayer: int
    price_usd: float
    team_size: int
    has_achievements: int

class PromptRequest(BaseModel):
    description: str = Field(
        ...,
        json_schema_extra={
            "example": "We are a small studio team of 4 devs making a $15 singleplayer indie RPG with steam achievements"
        }
    )

def parse_game_description(text: str) -> dict:
    text_lower = text.lower()

    genres = ['rpg', 'action', 'strategy', 'indie', 'shooter']
    found_genre = 'Action'
    for g in genres:
        if g in text_lower:
            found_genre = g.capitalize()
            break

    price_match = re.search(r'\$(\d+(?:\.\d+)?)', text)
    price = float(price_match.group(1)) if price_match else 20.0

    team_match = re.search(r'(\d+)\s*(?:devs|developers|team|people)', text_lower)
    team_size = int(team_match.group(1)) if team_match else 5

    is_multiplayer = 1 if 'multiplayer' in text_lower or 'co-op' in text_lower else 0
    has_achievements = 1 if 'achievement' in text_lower or 'trophies' in text_lower else 0

    return {
        "genre": found_genre,
        "is_multiplayer": is_multiplayer,
        "price_usd": price,
        "team_size": team_size,
        "has_achievements": has_achievements
    }

@app.get("/")
def home():
    return {"status": "online", "message": "ML Prediction API is active"}

@app.post("/predict")
def predict_rating(game: GameFeatures):
    input_data = pd.DataFrame([game.model_dump()])
    predicted_score = model.predict(input_data)[0]
    
    return {
        "genre": game.genre,
        "predicted_rating": round(float(predicted_score), 1),
        "verdict": "Overwhelmingly Positive" if predicted_score >= 85 else "Mostly Positive"
    }

@app.post("/predict-from-text")
def predict_from_text(request: PromptRequest):
    features = parse_game_description(request.description)
    
    input_data = pd.DataFrame([features])
    predicted_score = model.predict(input_data)[0]

    return {
        "raw_text_prompt": request.description,
        "parsed_features": features,
        "predicted_rating": round(float(predicted_score), 1),
        "verdict": "Overwhelmingly Positive" if predicted_score >= 85 else "Mostly Positive"
    }
