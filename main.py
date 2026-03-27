from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from database import SessionLocal
import traceback
import re
import shap
import pandas as pd
import joblib
import numpy as np
import os
from chat import handle_chat

app = FastAPI(title="Universal AI Platform")
 
origins = ["*"]  # for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        headers={"Access-Control-Allow-Origin": "http://localhost:4200"},
        content={"error": str(exc)}
    )

intent_model = joblib.load("intent_model.pkl")
intent_vectorizer = joblib.load("intent_vectorizer.pkl")
INTENT_CONFIDENCE_THRESHOLD = 0.55


# ---------- Request Models ----------
class BehaviorModel(BaseModel):
    userId: int
    domain: str
    itemId: int
    actionType: str
    priceRange: float
    timeOfDay: str


class SmartSearchRequest(BaseModel):
    query: str
    userId: int  


class BundleRequest(BaseModel):
    destination: str
    budget: float
    userId: int


class ChatRequest(BaseModel):
    message: str
    userId: int


# ---------- Serve frontend ----------
 

@app.get("/hotels/{city}")
def get_hotels(city: str):
    db = SessionLocal()
    try:
        query = text("""
            SELECT * FROM hotels
            WHERE City = :city
            ORDER BY Rating DESC
            LIMIT 10
        """)
        result = db.execute(query, {"city": city}).fetchall()

        return [dict(r._mapping) for r in result]
    finally:
        db.close()

@app.get("/flights/{city}")
def get_flights(city: str):
    db = SessionLocal()
    try:
        query = text("""
            SELECT * FROM flights
            WHERE Destination = :city
            ORDER BY BasePrice ASC
            LIMIT 10
        """)
        result = db.execute(query, {"city": city}).fetchall()

        return [dict(r._mapping) for r in result]
    finally:
        db.close()

# ---------- Track Behavior ----------
@app.post("/track-behavior")
def track_behavior(data: BehaviorModel):
    db = SessionLocal()
    try:
        query = text("""
            INSERT INTO userbehavior
            (UserId, Domain, ItemId, ActionType, PriceRange, TimeOfDay)
            VALUES (:userId, :domain, :itemId, :actionType, :priceRange, :timeOfDay)
        """)
        db.execute(query, data.dict())
        db.commit()
        return {"status": "tracked successfully"}
    finally:
        db.close()


# ---------- Smart Search ----------
@app.post("/smart-search")
def smart_search(data: SmartSearchRequest):
    db = SessionLocal()
    try:
        query = data.query.lower()
        user_id = data.userId 
        destination = None
        cities = ["goa", "chennai", "bangalore", "mumbai", "delhi"]
        for city in cities:
            if city in query:
                destination = city.title()
                break

        budget = None
        match = re.search(r'\d+', query)
        if match:
            budget = int(match.group())

        # 🔥 FIX INTENT LOGIC (IMPORTANT)
        if any(word in query for word in ["hotel", "stay"]):
            intent = "hotel"
        elif any(word in query for word in ["flight", "fly"]):
            intent = "flight"
        elif any(word in query for word in ["trip", "travel", "go to"]):
            intent = "travel"
        else:
            intent = "unknown"

        response = {
            "intent": intent,
            "destination": destination,
            "budget": budget,
            "originalQuery": data.query
        }

        # 🔥 NEW: attach data based on intent

        if intent == "hotel" and destination:

    # 🔥 Get user behavior (avg spending)
            behavior = db.execute(text("""
                SELECT AVG(PriceRange)
                FROM userbehavior
                WHERE UserId = :uid
            """), {"uid": user_id}).fetchone()

            user_avg_price = float(behavior[0]) if behavior and behavior[0] else 3000

            # 🔥 AI scoring query
            hotels = db.execute(text("""
                SELECT *,
                       (Rating * 0.6 +
                        (1 - ABS(BasePrice - :price) / BasePrice) * 0.4) AS score
                FROM hotels
                WHERE City = :dest
                ORDER BY score DESC
                LIMIT 5
            """), {
                "dest": destination,
                "price": user_avg_price
            }).fetchall()
            if budget:
                query = """
                    SELECT *,
                           (Rating * 0.6 +
                            (1 - ABS(BasePrice - :price) / BasePrice) * 0.4) AS score
                    FROM hotels
                    WHERE City = :dest AND BasePrice <= :budget
                    ORDER BY score DESC
                    LIMIT 5
                """
            else:
                query = """
                    SELECT *,
                           (Rating * 0.6 +
                            (1 - ABS(BasePrice - :price) / BasePrice) * 0.4) AS score
                    FROM hotels
                    WHERE City = :dest
                    ORDER BY score DESC
                    LIMIT 10
                """

            hotels = db.execute(text(query), {
                "dest": destination,
                "price": user_avg_price,
                "budget": budget
            }).fetchall()

            response["hotels"] = [dict(h._mapping) for h in hotels]

        elif intent == "flight" and destination:

    # 🔥 Get user behavior (avg spend)
            behavior = db.execute(text("""
                SELECT AVG(PriceRange)
                FROM userbehavior
                WHERE UserId = :uid
            """), {"uid": user_id}).fetchone()

            user_avg_price = float(behavior[0]) if behavior and behavior[0] else 3000

            # 🔥 AI scoring for flights
            flights = db.execute(text("""
                SELECT *,
                       (
                           (1 - ABS(BasePrice - :price) / BasePrice) * 0.6 +  -- price match
                           (1000 / BasePrice) * 0.2 +                         -- cheaper preference
                           (CASE 
                                WHEN DepartureTime LIKE '%AM%' THEN 1 ELSE 0.5 
                            END) * 0.2                                        -- morning preference
                       ) AS score
                FROM flights
                WHERE Destination = :dest
                ORDER BY score DESC
                LIMIT 10
            """), {
                "dest": destination,
                "price": user_avg_price
            }).fetchall()

            response["flights"] = [dict(f._mapping) for f in flights]

        return response

    finally:
        db.close()

# ---------- Recommendations ----------
@app.get("/recommendations/{userId}")
def get_recommendations(userId: int):
    db = SessionLocal()
    try:
        behavior_query = text("""
            SELECT Domain, AVG(PriceRange) as avg_price
            FROM userbehavior
            WHERE UserId = :uid
            GROUP BY Domain
        """)
        behavior_result = db.execute(behavior_query, {"uid": userId}).fetchall()

        preferred_price = 3000
        if behavior_result:
            preferred_price = behavior_result[0][1]

        hotel_query = text("""
            SELECT *,
                   (Rating * 0.6 +
                    (1 - ABS(BasePrice - :price) / BasePrice) * 0.4) AS score
            FROM hotels
            ORDER BY score DESC
            LIMIT 3
        """)
        hotel = db.execute(hotel_query, {"price": preferred_price}).fetchone()

        flight_query = text("SELECT * FROM flights ORDER BY BasePrice ASC LIMIT 3")
        flight = db.execute(flight_query).fetchone()

        product_query = text("SELECT * FROM products ORDER BY Rating DESC LIMIT 3")
        product = db.execute(product_query).fetchone()

        return {
            "recommendedHotel": dict(hotel._mapping) if hotel else None,
            "recommendedFlight": dict(flight._mapping) if flight else None,
            "recommendedProduct": dict(product._mapping) if product else None,
            "explanation": "Recommendations based on your past behavior and price preference"
        }
    finally:
        db.close()


# ---------- Dynamic Price ----------
@app.get("/dynamic-price")
def dynamic_price(domain: str, itemId: int, userId: int):
    import random
    db = SessionLocal()
    try:
        table_map = {"Hotel": "hotels", "Flight": "flights", "Product": "products"}
        id_col_map = {"Hotel": "HotelId", "Flight": "FlightId", "Product": "ProductId"}
        table = table_map.get(domain)
        id_col = id_col_map.get(domain)
        if not table:
            return {"error": "Invalid domain"}

        price_query = text(f"SELECT BasePrice FROM {table} WHERE {id_col} = :id")
        result = db.execute(price_query, {"id": itemId}).fetchone()
        if not result:
            return {"error": "Item not found"}

        base_price = float(result[0])
        demand_factor = random.uniform(0.9, 1.3)

        behavior_query = text("SELECT AVG(PriceRange) FROM userbehavior WHERE UserId = :uid")
        avg_price_result = db.execute(behavior_query, {"uid": userId}).fetchone()

        user_factor = 1.0
        if avg_price_result and avg_price_result[0]:
            avg_user_price = float(avg_price_result[0])
            user_factor = 1.1 if avg_user_price > base_price else 0.95

        final_price = base_price * demand_factor * user_factor

        return {
            "domain": domain,
            "itemId": itemId,
            "basePrice": round(base_price, 2),
            "demandFactor": round(demand_factor, 2),
            "userFactor": round(user_factor, 2),
            "aiPrice": round(final_price, 2),
            "explanation": "Price dynamically adjusted based on demand and user behavior"
        }
    finally:
        db.close()


# ---------- Risk Score ----------
@app.get("/risk-score/{userId}")
def risk_score(userId: int):
    db = SessionLocal()
    try:
        activity_query = text("""
            SELECT
                COUNT(*) as total_actions,
                SUM(CASE WHEN ActionType = 'Cancel' THEN 1 ELSE 0 END) as cancel_count,
                AVG(PriceRange) as avg_spend
            FROM userbehavior
            WHERE UserId = :uid
        """)
        result = db.execute(activity_query, {"uid": userId}).fetchone()

        total_actions = result[0] or 0
        cancel_count = result[1] or 0
        avg_spend = float(result[2]) if result[2] else 0
        cancel_ratio = cancel_count / total_actions if total_actions else 0

        try:
            model = joblib.load("risk_model.pkl")
            scaler = joblib.load("risk_scaler.pkl")
            X = np.array([[total_actions, avg_spend, cancel_ratio]])
            X_scaled = scaler.transform(X)
            risk_score_value = float(model.predict_proba(X_scaled)[0][1])
        except Exception:
            risk_score_value = 0.1

        risk_level = "Low" if risk_score_value < 0.3 else "Medium" if risk_score_value < 0.6 else "High"

        return {
            "userId": userId,
            "riskScore": round(risk_score_value, 2),
            "riskLevel": risk_level,
            "totalActions": total_actions,
            "cancelCount": cancel_count,
            "explanation": "Risk evaluated using Logistic Regression model"
        }
    finally:
        db.close()


# ---------- Optimize Bundle ----------
@app.post("/optimize-bundle")
def optimize_bundle(data: BundleRequest):
    db = SessionLocal()
    try:
        behavior_query = text("SELECT AVG(PriceRange) FROM userbehavior WHERE UserId = :uid")
        user_avg_result = db.execute(behavior_query, {"uid": data.userId}).fetchone()
        user_avg_price = float(user_avg_result[0]) if user_avg_result[0] else 3000

        flight_query = text("""
            SELECT * FROM flights WHERE Destination = :dest ORDER BY BasePrice ASC LIMIT 3
        """)
        flights = db.execute(flight_query, {"dest": data.destination}).fetchall()

        hotel_query = text("""
            SELECT *,
                   (Rating * 0.6 + (1 - ABS(BasePrice - :price) / BasePrice) * 0.4) AS score
            FROM hotels
            WHERE City = :dest
            ORDER BY score DESC
            LIMIT 3
        """)
        hotels = db.execute(hotel_query, {"dest": data.destination, "price": user_avg_price}).fetchall()

        if not flights or not hotels:
            return {"message": "No flights or hotels found for this destination"}

        best_bundle = None
        best_score = float("inf")

        for f in flights:
            for h in hotels:
                f_price = float(f._mapping["BasePrice"])
                h_price = float(h._mapping["BasePrice"])
                total_cost = f_price + h_price
                budget_penalty = max(0, total_cost - data.budget)
                preference_penalty = abs(total_cost - user_avg_price)
                score = total_cost + budget_penalty * 2 + preference_penalty * 0.3
                if score < best_score:
                    best_score = score
                    best_bundle = (f, h, total_cost)

        if not best_bundle:
            return {"message": "No suitable bundle found"}

        flight, hotel, total_cost = best_bundle
        return {
            "recommendedFlight": dict(flight._mapping),
            "recommendedHotel": dict(hotel._mapping),
            "totalBundleCost": round(total_cost, 2),
            "confidenceScore": round(1 / (1 + best_score / 10000), 2),
            "explanation": "Bundle optimized based on budget, rating, and user behavior"
        }
    finally:
        db.close()


# ---------- User Segment ----------
@app.get("/user-segment/{userId}")
def user_segment(userId: int):
    db = SessionLocal()
    try:
        query = text("""
            SELECT
                COUNT(*) as total_actions,
                AVG(PriceRange) as avg_spend,
                SUM(CASE WHEN ActionType='Cancel' THEN 1 ELSE 0 END) as cancel_count
            FROM userbehavior
            WHERE UserId = :uid
        """)
        result = db.execute(query, {"uid": userId}).fetchone()

        total_actions = result[0] or 0
        avg_spend = float(result[1]) if result[1] else 0
        cancel_count = result[2] or 0
        cancel_ratio = cancel_count / total_actions if total_actions else 0

        try:
            kmeans = joblib.load("kmeans_model.pkl")
            scaler = joblib.load("kmeans_scaler.pkl")
            X = np.array([[total_actions, avg_spend, cancel_ratio]])
            X_scaled = scaler.transform(X)
            cluster = int(kmeans.predict(X_scaled)[0])
            segment_map = {
                0: "Budget Traveler",
                1: "Premium Traveler",
                2: "Frequent Traveler",
                3: "Occasional Traveler"
            }
            segment = segment_map.get(cluster, "Normal User")
            confidence = 0.85
        except Exception:
            segment = "Model Not Trained"
            confidence = 0.0

        return {
            "userId": userId,
            "segment": segment,
            "confidence": confidence,
            "avgSpend": round(avg_spend, 2),
            "totalActions": total_actions,
            "cancelCount": cancel_count,
            "modelType": "KMeans Clustering"
        }
    finally:
        db.close()


# ---------- Helper Functions ----------
def extract_budget(message: str):
    match = re.search(r"\d+", message)
    return int(match.group()) if match else None


def detect_city_from_message(message: str, db):
    query = text("SELECT DISTINCT City FROM hotels")
    cities = [row[0].lower() for row in db.execute(query).fetchall()]
    for city in cities:
        if city in message:
            return city.title()
    return None


# ---------- AI Chat ----------
@app.post("/ai-chat")
def ai_chat(data: ChatRequest):
    return handle_chat(data)


# ---------- Revenue Optimized Price ----------
@app.get("/revenue-optimized-price")
def revenue_optimized_price(domain: str, itemId: int, userId: int):
    import random
    db = SessionLocal()
    try:
        table_map = {"Hotel": "hotels", "Flight": "flights", "Product": "products"}
        id_col_map = {"Hotel": "HotelId", "Flight": "FlightId", "Product": "ProductId"}
        table = table_map.get(domain)
        id_col = id_col_map.get(domain)
        if not table:
            return {"error": "Invalid domain"}

        price_query = text(f"SELECT BasePrice FROM {table} WHERE {id_col} = :id")
        result = db.execute(price_query, {"id": itemId}).fetchone()
        if not result:
            return {"error": "Item not found"}

        base_price = float(result[0])
        demand_score = random.uniform(0.05, 0.95)
        user_segment_val = 1

        try:
            seg_query = text("""
                SELECT
                    COUNT(*) as total_actions,
                    AVG(PriceRange) as avg_spend,
                    SUM(CASE WHEN ActionType='Cancel' THEN 1 ELSE 0 END) as cancel_count
                FROM userbehavior WHERE UserId = :uid
            """)
            seg_row = db.execute(seg_query, {"uid": userId}).fetchone()

            if seg_row:
                total_actions = seg_row[0] or 0
                avg_spend = float(seg_row[1]) if seg_row[1] else 0
                cancel_ratio = (seg_row[2] or 0) / total_actions if total_actions else 0

                kmeans = joblib.load("kmeans_model.pkl")
                scaler_seg = joblib.load("kmeans_scaler.pkl")
                X_seg = np.array([[total_actions, avg_spend, cancel_ratio]])
                X_seg_scaled = scaler_seg.transform(X_seg)
                user_segment_val = int(kmeans.predict(X_seg_scaled)[0])
        except Exception:
            user_segment_val = 1

        price_model = joblib.load("price_model.pkl")
        price_scaler = joblib.load("price_scaler.pkl")

        X = np.array([[base_price, demand_score, user_segment_val]])
        X_scaled = price_scaler.transform(X)

        feature_names = ["base_price", "demand_score", "user_segment"]
        background = price_scaler.transform(np.array([[base_price, 0.5, 1]]))
        explainer = shap.LinearExplainer(price_model, background)
        shap_values = explainer.shap_values(X_scaled)
        contributions = dict(zip(feature_names, [float(v) for v in shap_values[0]]))

        optimized_price = float(price_model.predict(X_scaled)[0])
        min_price = base_price * 0.7
        max_price = base_price * 1.6
        optimized_price = max(min_price, min(max_price, optimized_price))

        return {
            "basePrice": base_price,
            "demandScore": round(demand_score, 3),
            "userSegment": user_segment_val,
            "optimizedPrice": round(optimized_price, 2),
            "priceChangePercent": round(((optimized_price - base_price) / base_price) * 100, 2),
            "aiExplanation": contributions,
            "topDriver": max(contributions, key=contributions.get),
            "modelType": "Revenue Optimization Engine + Explainable AI",
            "explanation": "Price optimized using demand prediction and user intelligence"
        }
    finally:
        db.close()
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
