import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from sqlalchemy import text
from database import SessionLocal

# ===============================
# FEATURE EXTRACTION
# ===============================
def load_user_features():
    db = SessionLocal()

    query = text("""
        SELECT 
            UserId,
            COUNT(*) as total_actions,
            AVG(PriceRange) as avg_spend,
            SUM(CASE WHEN ActionType='Cancel' THEN 1 ELSE 0 END) as cancel_count
        FROM UserBehavior
        GROUP BY UserId
    """)

    rows = db.execute(query).fetchall()
    db.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "UserId", "total_actions", "avg_spend", "cancel_count"
    ])

    df["cancel_ratio"] = df["cancel_count"] / df["total_actions"].replace(0, 1)

    return df

# ===============================
# TRAIN KMEANS
# ===============================
def train_user_segmentation():
    df = load_user_features()
    if df.empty:
        return

    X = df[["total_actions", "avg_spend", "cancel_ratio"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(scaler, "kmeans_scaler.pkl")

# ===============================
# TRAIN RISK MODEL (Logistic)
# ===============================
def train_risk_model():
    df = load_user_features()
    if df.empty:
        print("❌ No data for risk model")
        return

    # ===============================
    # Create label
    # ===============================
    df["risk_label"] = (df["cancel_ratio"] > 0.4).astype(int)

    # 🚨 IMPORTANT FIX — ensure 2 classes
    if df["risk_label"].nunique() < 2:
        print("⚠️ Only one class found — injecting synthetic risky users")

        import numpy as np
        idx = np.random.choice(len(df), size=max(1, len(df)//10), replace=False)
        df.loc[idx, "risk_label"] = 1

    # ===============================
    # Prepare data
    # ===============================
    X = df[["total_actions", "avg_spend", "cancel_ratio"]]
    y = df["risk_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===============================
    # Train model
    # ===============================
    model = LogisticRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, "risk_model.pkl")
    joblib.dump(scaler, "risk_scaler.pkl")

    print("✅ Risk model trained successfully")
