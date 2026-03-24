from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from database import SessionLocal
import re
import shap
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI(title="Universal AI Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

intent_model = joblib.load("intent_model.pkl")
intent_vectorizer = joblib.load("intent_vectorizer.pkl")
INTENT_CONFIDENCE_THRESHOLD = 0.55


def predict_intent_with_confidence(message: str):
    X = intent_vectorizer.transform([message])
    probs = intent_model.predict_proba(X)[0]
    intent_index = np.argmax(probs)
    intent = intent_model.classes_[intent_index]
    confidence = float(probs[intent_index])
    return intent, confidence


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


class BundleRequest(BaseModel):
    destination: str
    budget: float
    userId: int


class ChatRequest(BaseModel):
    message: str
    userId: int


# ---------- Serve frontend ----------
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    
 
# ---------- Track Behavior ----------
@app.post("/track-behavior")
def track_behavior(data: BehaviorModel):
    db = SessionLocal()
    query = text("""
        INSERT INTO UserBehavior
        (UserId, Domain, ItemId, ActionType, PriceRange, TimeOfDay)
        VALUES (:userId, :domain, :itemId, :actionType, :priceRange, :timeOfDay)
    """)
    db.execute(query, data.dict())
    db.commit()
    db.close()
    return {"status": "tracked successfully"}


# ---------- Smart Search ----------
@app.post("/smart-search")
def smart_search(data: SmartSearchRequest):
    query = data.query.lower()

    destination = None
    cities = ["goa", "chennai", "bangalore", "mumbai", "delhi"]
    for city in cities:
        if city in query:
            destination = city.title()
            break

    budget = None
    budget_match = re.search(r'(\d+)', query)
    if budget_match:
        budget = int(budget_match.group(1))

    intent = "unknown"
    if destination:
        intent = "travel"
    elif any(word in query for word in ["trip", "travel", "flight", "go to"]):
        intent = "travel"
    elif any(word in query for word in ["hotel", "stay"]):
        intent = "hotel"
    elif any(word in query for word in ["buy", "product"]):
        intent = "shopping"

    return {
        "intent": intent,
        "destination": destination,
        "budget": budget,
        "originalQuery": data.query
    }


# ---------- Recommendations ----------
@app.get("/recommendations/{userId}")
def get_recommendations(userId: int):
    db = SessionLocal()

    behavior_query = text("""
        SELECT Domain, AVG(PriceRange) as avg_price
        FROM UserBehavior
        WHERE UserId = :uid
        GROUP BY Domain
    """)
    behavior_result = db.execute(behavior_query, {"uid": userId}).fetchall()

    preferred_domain = None
    preferred_price = 3000

    if behavior_result:
        preferred_domain = behavior_result[0][0]
        preferred_price = behavior_result[0][1]

    hotel_query = text("""
        SELECT *,
               (Rating * 0.6 +
                (1 - ABS(BasePrice - :price) / BasePrice) * 0.4) AS score
        FROM Hotels
        ORDER BY score DESC
        LIMIT 1
    """)
    hotel = db.execute(hotel_query, {"price": preferred_price}).fetchone()

    flight_query = text("SELECT * FROM Flights ORDER BY BasePrice ASC LIMIT 3")
    flight = db.execute(flight_query).fetchone()

    product_query = text("SELECT * FROM Products ORDER BY Rating DESC LIMIT 3")
    product = db.execute(product_query).fetchone()

    db.close()

    return {
        "recommendedHotel": dict(hotel._mapping) if hotel else None,
        "recommendedFlight": dict(flight._mapping) if flight else None,
        "recommendedProduct": dict(product._mapping) if product else None,
        "explanation": "Recommendations based on your past behavior and price preference"
    }


# ---------- Dynamic Price ----------
@app.get("/dynamic-price")
def dynamic_price(domain: str, itemId: int, userId: int):
    import random
    db = SessionLocal()

    table_map = {"Hotel": "Hotels", "Flight": "Flights", "Product": "Products"}
    table = table_map.get(domain)
    if not table:
        return {"error": "Invalid domain"}

    price_query = text(f"SELECT BasePrice FROM {table} WHERE {table[:-1]}Id = :id")
    result = db.execute(price_query, {"id": itemId}).fetchone()
    if not result:
        db.close()
        return {"error": "Item not found"}

    base_price = float(result[0])
    demand_factor = random.uniform(0.9, 1.3)

    behavior_query = text("SELECT AVG(PriceRange) FROM UserBehavior WHERE UserId = :uid")
    avg_price_result = db.execute(behavior_query, {"uid": userId}).fetchone()

    user_factor = 1.0
    if avg_price_result and avg_price_result[0]:
        avg_user_price = float(avg_price_result[0])
        user_factor = 1.1 if avg_user_price > base_price else 0.95

    final_price = base_price * demand_factor * user_factor
    db.close()

    return {
        "domain": domain,
        "itemId": itemId,
        "basePrice": round(base_price, 2),
        "demandFactor": round(demand_factor, 2),
        "userFactor": round(user_factor, 2),
        "aiPrice": round(final_price, 2),
        "explanation": "Price dynamically adjusted based on demand and user behavior"
    }


# ---------- Risk Score ----------
@app.get("/risk-score/{userId}")
def risk_score(userId: int):
    db = SessionLocal()

    activity_query = text("""
        SELECT
            COUNT(*) as total_actions,
            SUM(CASE WHEN ActionType = 'Cancel' THEN 1 ELSE 0 END) as cancel_count,
            AVG(PriceRange) as avg_spend
        FROM UserBehavior
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
    db.close()

    return {
        "userId": userId,
        "riskScore": round(risk_score_value, 2),
        "riskLevel": risk_level,
        "totalActions": total_actions,
        "cancelCount": cancel_count,
        "explanation": "Risk evaluated using Logistic Regression model"
    }


# ---------- Optimize Bundle ----------
@app.post("/optimize-bundle")
def optimize_bundle(data: BundleRequest):
    db = SessionLocal()

    behavior_query = text("SELECT AVG(PriceRange) FROM UserBehavior WHERE UserId = :uid")
    user_avg_result = db.execute(behavior_query, {"uid": data.userId}).fetchone()
    user_avg_price = float(user_avg_result[0]) if user_avg_result[0] else 3000

    flight_query = text("""
        SELECT * FROM Flights WHERE Destination = :dest ORDER BY BasePrice ASC LIMIT 3
    """)
    flights = db.execute(flight_query, {"dest": data.destination}).fetchall()

    hotel_query = text("""
        SELECT *,
               (Rating * 0.6 + (1 - ABS(BasePrice - :price) / BasePrice) * 0.4) AS score
        FROM Hotels
        WHERE City = :dest
        ORDER BY score DESC
        LIMIT 3
    """)
    hotels = db.execute(hotel_query, {"dest": data.destination, "price": user_avg_price}).fetchall()

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

    db.close()

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


# ---------- User Segment ----------
@app.get("/user-segment/{userId}")
def user_segment(userId: int):
    db = SessionLocal()

    query = text("""
        SELECT
            COUNT(*) as total_actions,
            AVG(PriceRange) as avg_spend,
            SUM(CASE WHEN ActionType='Cancel' THEN 1 ELSE 0 END) as cancel_count
        FROM UserBehavior
        WHERE UserId = :uid
    """)
    result = db.execute(query, {"uid": userId}).fetchone()
    db.close()

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


# ---------- Helper Functions ----------
def extract_budget(message: str):
    match = re.search(r"\d+", message)
    return int(match.group()) if match else None


def detect_city_from_message(message: str, db):
    query = text("SELECT DISTINCT City FROM Hotels")
    cities = [row[0].lower() for row in db.execute(query).fetchall()]
    for city in cities:
        if city in message:
            return city.title()
    return None


# ---------- AI Chat ----------
@app.post("/ai-chat")
def ai_chat(data: ChatRequest):
    db = SessionLocal()

    msg = data.message.lower().strip()
    detected_city   = detect_city_from_message(msg, db)
    detected_budget = extract_budget(msg)

    # ---------- Load session ----------
    session_query = text("""
        SELECT LastIntent, LastDestination, LastQuestion, LastBudget
        FROM ChatSession WHERE UserId = :uid
    """)
    session_row = db.execute(session_query, {"uid": data.userId}).fetchone()

    last_intent      = session_row[0] if session_row else None
    last_destination = session_row[1] if session_row else None
    last_question    = session_row[2] if session_row else None
    last_budget      = float(session_row[3]) if (session_row and session_row[3]) else None

    # =================================================
    # 🔥 STATE MACHINE FIRST
    # =================================================

    # STEP: travel flow — waiting for budget
    if last_question == "ask_budget":
        if detected_budget:
            response  = f"Got it! 💰 ₹{detected_budget} budget noted. Do you prefer morning or evening travel?"
            new_state = "ask_time"
            db.execute(text("""
                UPDATE ChatSession SET LastQuestion='ask_time', LastBudget=:budget WHERE UserId=:uid
            """), {"uid": data.userId, "budget": detected_budget})
        else:
            response  = "Please share a budget amount, e.g. '5000' or '10000'."
            new_state = "ask_budget"

        db.commit()
        db.close()
        return {
            "reply": response, "intentDetected": last_intent, "confidence": 1.0,
            "conversationState": new_state, "modelType": "State Machine",
            "fallbackTriggered": False, "recommendations": None, "bundle": None, "pricing": None,
        }

    # STEP: travel flow — waiting for city
    if last_question == "ask_city":
        if detected_city:
            response  = f"📍 {detected_city} it is! What is your budget? (e.g. 10000)"
            new_state = "ask_budget"
            db.execute(text("""
                UPDATE ChatSession SET LastDestination=:city, LastQuestion='ask_budget' WHERE UserId=:uid
            """), {"uid": data.userId, "city": detected_city})
        else:
            response  = "I couldn't detect a city. Try: Goa, Mumbai, Delhi, Chennai, or Bangalore."
            new_state = "ask_city"

        db.commit()
        db.close()
        return {
            "reply": response, "intentDetected": last_intent, "confidence": 1.0,
            "conversationState": new_state, "modelType": "State Machine",
            "fallbackTriggered": False, "recommendations": None, "bundle": None, "pricing": None,
        }

    # STEP: hotel flow — waiting for city
    if last_question == "ask_hotel_details":
        if detected_city:
            response  = f"Great! 🏨 Found hotels in {detected_city}. What's your budget? (e.g. 5000)"
            new_state = "ask_hotel_budget"
            db.execute(text("""
                INSERT INTO ChatSession (UserId, LastIntent, LastDestination, LastQuestion, LastBudget)
                VALUES (:uid, 'hotel', :city, 'ask_hotel_budget', NULL)
                ON DUPLICATE KEY UPDATE
                    LastIntent='hotel', LastDestination=:city,
                    LastQuestion='ask_hotel_budget', LastBudget=NULL
            """), {"uid": data.userId, "city": detected_city})
        else:
            response  = "Please tell me a city name — Goa, Chennai, Mumbai, Delhi, or Bangalore."
            new_state = "ask_hotel_details"

        db.commit()
        db.close()
        return {
            "reply": response, "intentDetected": last_intent, "confidence": 1.0,
            "conversationState": new_state, "modelType": "State Machine",
            "fallbackTriggered": False, "recommendations": None, "bundle": None, "pricing": None,
        }

    # STEP: hotel flow — waiting for budget → show hotel results
    if last_question == "ask_hotel_budget":
        if detected_budget:
            destination = last_destination or "Chennai"
            budget      = detected_budget

            db.execute(text("""
                UPDATE ChatSession SET LastQuestion='completed', LastBudget=:budget WHERE UserId=:uid
            """), {"uid": data.userId, "budget": budget})

            recommendations = None
            try:
                hotel_q = text("""
                    SELECT *, (Rating*0.6 + (1 - ABS(BasePrice - :price)/BasePrice)*0.4) AS score
                    FROM Hotels WHERE City = :dest ORDER BY score DESC LIMIT 1
                """)
                hotel_rec = db.execute(hotel_q, {"dest": destination, "price": budget}).fetchone()
                recommendations = {
                    "hotel":  dict(hotel_rec._mapping) if hotel_rec else None,
                    "flight": None,
                }
            except Exception:
                recommendations = None

            pricing = None
            try:
                import random
                p_res = db.execute(text("""
                    SELECT BasePrice FROM Hotels WHERE City=:dest ORDER BY Rating DESC LIMIT 1
                """), {"dest": destination}).fetchone()
                base   = float(p_res[0]) if p_res else budget
                demand = round(random.uniform(0.9, 1.3), 2)
                ai_p   = round(base * demand, 2)
                pricing = {
                    "basePrice": base, "demandFactor": demand, "aiPrice": ai_p,
                    "change": round(((ai_p - base) / base) * 100, 1),
                }
            except Exception:
                pricing = None

            db.commit()
            db.close()
            return {
                "reply": f"🏨 Here are the best hotels in {destination} within ₹{budget} budget:",
                "intentDetected": last_intent, "confidence": 1.0,
                "conversationState": "completed", "modelType": "State Machine",
                "fallbackTriggered": False, "recommendations": recommendations,
                "bundle": None, "pricing": pricing,
            }
        else:
            db.commit()
            db.close()
            return {
                "reply": "Please share your budget amount, e.g. '5000' or '10000'.",
                "intentDetected": last_intent, "confidence": 1.0,
                "conversationState": "ask_hotel_budget", "modelType": "State Machine",
                "fallbackTriggered": False, "recommendations": None, "bundle": None, "pricing": None,
            }

    # STEP: travel flow — waiting for time of day → show full results
    if last_question == "ask_time":
        if any(t in msg for t in ["morning", "evening", "afternoon", "night"]):
            new_state   = "completed"
            destination = last_destination or "Goa"
            budget      = last_budget or 10000

            db.execute(text("""
                UPDATE ChatSession SET LastQuestion='completed' WHERE UserId=:uid
            """), {"uid": data.userId})

            recommendations = None
            try:
                behavior_q      = text("SELECT AVG(PriceRange) FROM UserBehavior WHERE UserId=:uid")
                b_res           = db.execute(behavior_q, {"uid": data.userId}).fetchone()
                preferred_price = float(b_res[0]) if b_res and b_res[0] else 3000
                hotel_q         = text("""
                    SELECT *, (Rating*0.6 + (1-ABS(BasePrice-:price)/BasePrice)*0.4) AS score
                    FROM Hotels ORDER BY score DESC LIMIT 1
                """)
                hotel_rec  = db.execute(hotel_q, {"price": preferred_price}).fetchone()
                flight_q   = text("SELECT * FROM Flights ORDER BY BasePrice ASC LIMIT 1")
                flight_rec = db.execute(flight_q).fetchone()
                recommendations = {
                    "hotel":  dict(hotel_rec._mapping)  if hotel_rec  else None,
                    "flight": dict(flight_rec._mapping) if flight_rec else None,
                }
            except Exception:
                recommendations = None

            bundle = None
            try:
                flights = db.execute(text("""
                    SELECT * FROM Flights WHERE Destination=:dest ORDER BY BasePrice ASC LIMIT 3
                """), {"dest": destination}).fetchall()
                hotels = db.execute(text("""
                    SELECT *, (Rating*0.6+(1-ABS(BasePrice-:price)/BasePrice)*0.4) AS score
                    FROM Hotels WHERE City=:dest ORDER BY score DESC LIMIT 3
                """), {"dest": destination, "price": budget / 2}).fetchall()

                best_bundle = None
                best_score  = float("inf")
                for f in flights:
                    for h in hotels:
                        fp    = float(f._mapping["BasePrice"])
                        hp    = float(h._mapping["BasePrice"])
                        total = fp + hp
                        score = total + max(0, total - budget) * 2
                        if score < best_score:
                            best_score  = score
                            best_bundle = (f, h, total)

                if best_bundle:
                    f, h, total = best_bundle
                    bundle = {
                        "flight":     dict(f._mapping),
                        "hotel":      dict(h._mapping),
                        "totalCost":  round(total, 2),
                        "confidence": round(1 / (1 + best_score / 10000), 2),
                    }
            except Exception:
                bundle = None

            pricing = None
            try:
                import random
                p_res = db.execute(text("""
                    SELECT BasePrice FROM Hotels WHERE City=:dest ORDER BY Rating DESC LIMIT 1
                """), {"dest": destination}).fetchone()
                if not p_res:
                    p_res = db.execute(text("SELECT BasePrice FROM Hotels ORDER BY Rating DESC LIMIT 1")).fetchone()
                base   = float(p_res[0]) if p_res else 4500
                demand = round(random.uniform(0.9, 1.3), 2)
                ai_p   = round(base * demand, 2)
                pricing = {
                    "basePrice": base, "demandFactor": demand, "aiPrice": ai_p,
                    "change": round(((ai_p - base) / base) * 100, 1),
                }
            except Exception:
                pricing = None

            db.commit()
            db.close()
            return {
                "reply":             f"🎉 Perfect! Here's your complete travel plan for {destination} within ₹{int(budget)} budget:",
                "intentDetected":    last_intent, "confidence": 1.0,
                "conversationState": new_state, "modelType": "State Machine",
                "fallbackTriggered": False, "recommendations": recommendations,
                "bundle": bundle, "pricing": pricing,
            }
        else:
            db.commit()
            db.close()
            return {
                "reply": "Do you prefer morning or evening travel?",
                "intentDetected": last_intent, "confidence": 1.0,
                "conversationState": "ask_time", "modelType": "State Machine",
                "fallbackTriggered": False, "recommendations": None, "bundle": None, "pricing": None,
            }

    # =================================================
    # 🤖 INTENT MODEL — fresh messages only
    # =================================================
    intent, confidence = predict_intent_with_confidence(msg)

    if confidence < INTENT_CONFIDENCE_THRESHOLD:
        db.close()
        return {
            "reply": "I'm not fully sure what you mean. Are you looking for travel, hotel?",
            "intentDetected": intent, "confidence": round(confidence, 2),
            "conversationState": "unknown", "modelType": "ML Intent Classifier",
            "fallbackTriggered": True, "recommendations": None, "bundle": None, "pricing": None,
        }

    response  = "I'm your AI travel assistant."
    new_state = "new"

    if intent == "greeting":
        response  = "Hello! 👋 I can help you plan trips, find hotels, and get the best prices. Which city are you planning to visit?"
        new_state = "greeted"

    elif intent == "travel" and detected_city:
        response  = f"Great choice! ✈️ What is your budget for the {detected_city} trip? (e.g. 10000)"
        new_state = "ask_budget"
        db.execute(text("""
            INSERT INTO ChatSession (UserId, LastIntent, LastDestination, LastQuestion, LastBudget)
            VALUES (:uid, 'travel', :city, 'ask_budget', NULL)
            ON DUPLICATE KEY UPDATE
                LastIntent='travel', LastDestination=:city,
                LastQuestion='ask_budget', LastBudget=NULL
        """), {"uid": data.userId, "city": detected_city})

    elif intent == "travel" and not detected_city:
        response  = "Sure! ✈️ Which city are you planning to visit? (Goa, Chennai, Mumbai, Delhi, Bangalore)"
        new_state = "ask_city"
        db.execute(text("""
            INSERT INTO ChatSession (UserId, LastIntent, LastDestination, LastQuestion, LastBudget)
            VALUES (:uid, 'travel', NULL, 'ask_city', NULL)
            ON DUPLICATE KEY UPDATE
                LastIntent='travel', LastDestination=NULL,
                LastQuestion='ask_city', LastBudget=NULL
        """), {"uid": data.userId})

    elif intent == "hotel":
        if detected_city and detected_budget:
            response  = f"🏨 Got it! Finding best hotels in {detected_city} within ₹{detected_budget}."
            new_state = "ask_hotel_budget"
            db.execute(text("""
                INSERT INTO ChatSession (UserId, LastIntent, LastDestination, LastQuestion, LastBudget)
                VALUES (:uid, 'hotel', :city, 'ask_hotel_budget', :budget)
                ON DUPLICATE KEY UPDATE
                    LastIntent='hotel', LastDestination=:city,
                    LastQuestion='ask_hotel_budget', LastBudget=:budget
            """), {"uid": data.userId, "city": detected_city, "budget": detected_budget})
        elif detected_city:
            response  = f"🏨 {detected_city} noted! What's your budget? (e.g. 5000)"
            new_state = "ask_hotel_budget"
            db.execute(text("""
                INSERT INTO ChatSession (UserId, LastIntent, LastDestination, LastQuestion, LastBudget)
                VALUES (:uid, 'hotel', :city, 'ask_hotel_budget', NULL)
                ON DUPLICATE KEY UPDATE
                    LastIntent='hotel', LastDestination=:city,
                    LastQuestion='ask_hotel_budget', LastBudget=NULL
            """), {"uid": data.userId, "city": detected_city})
        else:
            response  = "Which city do you need a hotel in? (Goa, Chennai, Mumbai, Delhi, Bangalore)"
            new_state = "ask_hotel_details"
            db.execute(text("""
                INSERT INTO ChatSession (UserId, LastIntent, LastDestination, LastQuestion, LastBudget)
                VALUES (:uid, 'hotel', NULL, 'ask_hotel_details', NULL)
                ON DUPLICATE KEY UPDATE
                    LastIntent='hotel', LastDestination=NULL,
                    LastQuestion='ask_hotel_details', LastBudget=NULL
            """), {"uid": data.userId})

    elif intent == "shopping":
        response = "🛍️ What are you looking to buy?"
        new_state = "ask_shopping_details"

        db.execute(text("""
            INSERT INTO ChatSession (UserId, LastIntent, LastDestination, LastQuestion, LastBudget)
            VALUES (:uid, 'shopping', NULL, 'ask_shopping_details', NULL)
            ON DUPLICATE KEY UPDATE
                LastIntent='shopping',
                LastDestination=NULL,
                LastQuestion='ask_shopping_details',
                LastBudget=NULL
        """), {"uid": data.userId})

    # ==============================
    # 7. FALLBACK
    # ==============================
    else:
        response = "🤖 I can help with travel, hotels, or shopping. What do you need?"
        new_state = "unknown"
        

    db.commit()
    db.close()

    return {
        "reply": response,
        "intentDetected": intent,
        "confidence": round(confidence, 2),
        "conversationState": new_state,
        "modelType": "ML Intent Classifier",
        "fallbackTriggered": False,
        "recommendations": None,
        "bundle": None,
        "pricing": None,
    }


# ---------- Revenue Optimized Price ----------
@app.get("/revenue-optimized-price")
def revenue_optimized_price(domain: str, itemId: int, userId: int):
    import random
    db = SessionLocal()

    table_map = {"Hotel": "Hotels", "Flight": "Flights", "Product": "Products"}
    table = table_map.get(domain)
    if not table:
        return {"error": "Invalid domain"}

    price_query = text(f"SELECT BasePrice FROM {table} WHERE {table[:-1]}Id = :id")
    result = db.execute(price_query, {"id": itemId}).fetchone()
    if not result:
        db.close()
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
            FROM UserBehavior WHERE UserId = :uid
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

    db.close()

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
