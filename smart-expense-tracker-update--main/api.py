from fastapi import FastAPI
from pydantic import BaseModel
import joblib # type: ignore

from pipeline import build_features, generate_final_recommendation, prepare_for_clustering # type: ignore

app = FastAPI()

# load models
scaler = joblib.load("robust_scaler.joblib")
kmeans = joblib.load("kmeans_model.joblib")
cluster_baseline = joblib.load("cluster_baseline.joblib")

# input schema
class UserInput(BaseModel):
    salary: float
    food: float
    drink: float
    shopping: float
    transport: float
    bills: float
    health: float
    entertainment: float


@app.get("/")
def home():
    return {"message": "Smart Expense API 🚀"}


@app.post("/analyze")
def analyze(data: UserInput):

    spend_dict = {
        "food": data.food,
        "drink": data.drink,
        "shopping": data.shopping,
        "transport": data.transport,
        "bills": data.bills,
        "health": data.health,
        "entertainment": data.entertainment
    }

    total_spend = sum(spend_dict.values())
    remaining = data.salary - total_spend

    # نفس logic بتاعك بالظبط
    new_user_df = build_features(data.salary, spend_dict)
    X_new = prepare_for_clustering(new_user_df)
    X_scaled = scaler.transform(X_new)
    cluster = int(kmeans.predict(X_scaled)[0])
    new_user_df['cluster'] = cluster

    recommendation = generate_final_recommendation(
        new_user_df.iloc[0],
        cluster_baseline
    )

    return {
        "total_spend": total_spend,
        "remaining": remaining,
        "cluster": cluster,
        "recommendation": recommendation
    }