import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def train_demand_model():
    # 🔥 synthetic training data (you can later replace with real logs)
    np.random.seed(42)

    data_size = 500

    df = pd.DataFrame({
        "search_count": np.random.randint(10, 500, data_size),
        "booking_count": np.random.randint(1, 200, data_size),
        "season_factor": np.random.uniform(0.5, 1.5, data_size),
    })

    # demand formula (hidden truth for training)
    df["demand_score"] = (
        df["search_count"] * 0.4 +
        df["booking_count"] * 0.5
    ) * df["season_factor"] / 1000

    X = df[["search_count", "booking_count", "season_factor"]]
    y = df["demand_score"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, "demand_model.pkl")
    joblib.dump(scaler, "demand_scaler.pkl")

    print("✅ Demand model trained")

if __name__ == "__main__":
    train_demand_model()
