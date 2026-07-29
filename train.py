import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

np.random.seed(42)
n_samples = 2000

genres = np.random.choice(['Action', 'RPG', 'Strategy', 'Indie', 'Shooter'], n_samples)
is_multiplayer = np.random.choice([0, 1], n_samples)
price_usd = np.random.uniform(0, 100, n_samples)
price_usd[:100] = np.random.uniform(100, 300, 100) 

team_size = np.random.randint(1, 200, n_samples)
has_achievements = np.random.choice([0, 1], n_samples)

data = pd.DataFrame({
    'genre': genres,
    'is_multiplayer': is_multiplayer,
    'price_usd': price_usd,
    'team_size': team_size,
    'has_achievements': has_achievements,
})

base_score = 65.0

greed_penalty = np.where((data['price_usd'] > 30) & (data['team_size'] <= 3), -35.0, 0.0)
overprice_penalty = np.where(data['price_usd'] > 80, -0.4 * (data['price_usd'] - 80), 0.0)
team_bonus = np.log1p(data['team_size']) * 4.0

genre_bonus = (
    (data['genre'] == 'RPG') * 5 + 
    (data['genre'] == 'Strategy') * 3 + 
    (data['genre'] == 'Indie') * 2
)
feature_bonus = (data['is_multiplayer'] * 4) + (data['has_achievements'] * 3)

raw_rating = (
    base_score 
    + genre_bonus 
    + feature_bonus 
    + team_bonus 
    + greed_penalty 
    + overprice_penalty 
    + np.random.normal(0, 4, n_samples)
)

data['rating'] = np.clip(raw_rating, 10, 100)

X = data.drop(columns=['rating'])
y = data['rating']

categorical_cols = ['genre']
numeric_cols = ['is_multiplayer', 'price_usd', 'team_size', 'has_achievements']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

joblib.dump(model_pipeline, 'game_rating_model.joblib')
print("Model saved as 'game_rating_model.joblib'")
