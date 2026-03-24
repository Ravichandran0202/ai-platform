import pandas as pd
import numpy as np
import joblib
from sqlalchemy import text
from database import SessionLocal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def train_price_model():
    db = SessionLocal()

    # 🔥 Load hotel price data
    query = text("""
        SELECT BasePrice, Rating
        FROM Hotels
        WHERE BasePrice IS NOT NULL
    """)

    rows = db.execute(query).fetchall()
    db.close()

    if not rows:
        print("❌ No hotel data found")
        return

    df = pd.DataFrame(rows, columns=["BasePrice", "Rating"])

    # ✅ VERY IMPORTANT — fix Decimal issue
    df["BasePrice"] = df["BasePrice"].astype(float)
    df["Rating"] = df["Rating"].astype(float)

    # 🔥 Create synthetic demand feature
    np.random.seed(42)
    df["demand"] = np.random.uniform(0.8, 1.3, len(df))

    # 🔥 Target price (simulated smart price)
    df["target_price"] = (
        df["BasePrice"] * (1 + (df["Rating"] - 3) * 0.05) * df["demand"]
    )

    X = df[["BasePrice", "Rating", "demand"]]
    y = df["target_price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, "price_model.pkl")
    joblib.dump(scaler, "price_scaler.pkl")

    print("✅ Price intelligence model trained successfully")


if __name__ == "__main__":
    print("🚀 Training price model...")
    train_price_model()
