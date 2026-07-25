# 🎮 Game Analytics ML API

A high-performance Machine Learning REST API built with **FastAPI**, **Scikit-Learn**, and **Pydantic**. This service predicts expected Metacritic ratings and overall market reception for video games based on attributes such as genre, pricing, team size, and feature support.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
---

## 🚀 Features

- 🤖 **Predictive ML Pipeline:** Powered by a `RandomForestRegressor` model wrapped in a Scikit-Learn pipeline.
- 📝 **Natural Language Parsing:** Send plain-English descriptions of your game idea, and the API automatically extracts relevant features.
- ⚡ **RESTful Endpoints:** Fast JSON-based prediction server with built-in validation using Pydantic.
- 📖 **Interactive API Docs:** Automatic Swagger UI (`/docs`) and ReDoc (`/redoc`).

---

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **Server:** Uvicorn
- **Machine Learning:** Scikit-Learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Validation:** Pydantic

---

# 📦 Local Setup & Installation

## Prerequisites

- Python 3.9+

---

## 1. Clone the Repository

```bash
git clone https://github.com/raveendra-7/game-analytics-ml-api.git
cd game-analytics-ml-api
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Train the Machine Learning Model

```bash
python train.py
```

This generates:

```
game_rating_model.joblib
```

---

## 4. Start the API Server

```bash
uvicorn app:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

---

# 📖 API Documentation

Once the server is running:

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

# 📡 API Reference

## 1. Predict from Natural Language

### Endpoint

```http
POST /predict-from-text
```

### Request

```json
{
  "description": "We are a small studio team of 4 developers making a $15 single-player indie RPG with Steam achievements."
}
```

### Example Python Client

```python
import requests

url = "http://127.0.0.1:8000/predict-from-text"

payload = {
    "description": "We are a small team of 4 developers building a $15 indie RPG with Steam achievements and multiplayer support."
}

response = requests.post(url, json=payload)

print(response.json())
```

### Example Response

```json
{
  "raw_text_prompt": "We are a small team of 4 developers building a $15 indie RPG with Steam achievements and multiplayer support.",
  "parsed_features": {
    "genre": "RPG",
    "is_multiplayer": 1,
    "price_usd": 15.0,
    "team_size": 4,
    "has_achievements": 1
  },
  "predicted_rating": 89.2,
  "verdict": "Overwhelmingly Positive"
}
```

---

## 2. Predict from Structured Features

### Endpoint

```http
POST /predict
```

### Request

```json
{
  "genre": "RPG",
  "is_multiplayer": 1,
  "price_usd": 29.99,
  "team_size": 15,
  "has_achievements": 1
}
```

### Example Response

```json
{
  "genre": "RPG",
  "predicted_rating": 88.4,
  "verdict": "Overwhelmingly Positive"
}
```

---

# 📂 Project Structure

```text
game-analytics-ml-api/
│
├── app.py                   # FastAPI application with NLP parser
├── train.py                 # Model training script
├── game_rating_model.joblib # Trained ML model
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

# 🚀 Future Improvements

- Docker support
- Batch prediction endpoint
- Model versioning
- Cloud deployment (Render, Railway, Azure)
- Authentication & rate limiting
- Improved NLP feature extraction
- Additional game metadata support

---

# 👨‍💻 Author

**Raveendra Gunapu**

- GitHub: https://github.com/raveendra-7

---

## 📄 License

This project is intended for educational and portfolio purposes.