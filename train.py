import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'genre': np.random.choice(['Action', 'RPG', 'Strategy', 'Indie', 'Shooter'], n_samples),
    'is_multiplayer': np.random.choice([0, 1], n_samples),
    'price_usd': np.random.uniform(0, 70, n_samples),
    'team_size': np.random.randint(1, 200, n_samples),
    'has_achievements': np.random.choice([0, 1], n_samples),
})

data['rating'] = (
    70 
    + (data['genre'] == 'RPG') * 8 
    + (data['price_usd'] * 0.15) 
    + (data['is_multiplayer'] * 5) 
    + np.random.normal(0, 5, n_samples)
).clip(40, 100)

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
print("Model trained and saved as 'game_rating_model.joblib'")