import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from database import SessionLocal

def train_hotel_ranker():
    db = SessionLocal()

    query = text("""
        SELECT 
            h.HotelId,
            h.BasePrice,
            h.Rating,
            h.City,
            COALESCE(AVG(ub.PriceRange), 3000) as user_avg_spend
        FROM Hotels h
        LEFT JOIN UserBehavior ub ON 1=1
        GROUP BY h.HotelId, h.BasePrice, h.Rating, h.City
    """)

    rows = db.execute(query).fetchall()
    db.close()

    df = pd.DataFrame(rows, columns=[
        "HotelId", "BasePrice", "Rating", "City", "user_avg_spend"
    ])

    # 🎯 synthetic target (demo but ML-valid)
    df["target_score"] = (
        df["Rating"] * 0.6 +
        (1 - abs(df["BasePrice"] - df["user_avg_spend"]) / df["BasePrice"]) * 0.4
    )

    X = df[["BasePrice", "Rating", "user_avg_spend"]]
    y = df["target_score"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, "hotel_ranker.pkl")
    joblib.dump(scaler, "hotel_ranker_scaler.pkl")

    print("✅ Hotel ranker trained")

if __name__ == "__main__":
    train_hotel_ranker()
