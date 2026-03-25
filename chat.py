import re
import random
from sqlalchemy import text
from database import SessionLocal
from ml_utils import predict_intent_with_confidence

INTENT_CONFIDENCE_THRESHOLD = 0.55
CITIES = ["goa", "mumbai", "delhi", "chennai", "bangalore"]


# ── extractors ────────────────────────────────

def detect_city(msg: str):
    for c in CITIES:
        if c in msg:
            return c.capitalize()
    return None


def extract_budget(msg: str):
    m = re.search(r"\d{3,6}", msg)
    return float(m.group()) if m else None


def extract_date(msg: str):
    m = re.search(
        r"(\d{1,2}[\/\-]\d{1,2}[\/\-]?\d{0,4}"
        r"|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
        r"|next\s+\w+|this\s+\w+|\d+\s+days?)",
        msg, re.I,
    )
    return m.group() if m else None


def extract_rating(msg: str):
    m = (
        re.search(r"([345])\s*star", msg)
        or re.search(r"([345])\s*\+", msg)
        or re.search(r"^([345])$", msg.strip())
    )
    return int(m.group(1)) if m else None


def extract_timing(msg: str):
    if any(w in msg for w in ["morning", "early", "6am", "7am", "8am", "9am"]):
        return "morning"
    if any(w in msg for w in ["afternoon", "12pm", "1pm", "2pm", "3pm"]):
        return "afternoon"
    if any(w in msg for w in ["evening", "4pm", "5pm", "6pm", "7pm", "8pm"]):
        return "evening"
    if any(w in msg for w in ["night", "9pm", "10pm", "11pm"]):
        return "night"
    return None


# ── DB queries ────────────────────────────────

def get_hotels(db, city: str, budget: float, rating):
    q = """
        SELECT HotelId, HotelName, City, BasePrice, Rating, AvailableRooms,
               (Rating*0.6 + (1-ABS(BasePrice-:price)/BasePrice)*0.4) AS score
        FROM hotels WHERE City=:city
    """
    params = {"city": city, "price": budget}
    if rating:
        q += " AND Rating >= :rating"
        params["rating"] = rating
    q += " ORDER BY score DESC LIMIT 3"
    rows = db.execute(text(q), params).fetchall()
    return [dict(r._mapping) for r in rows]


def get_flights(db, destination: str, timing):
    time_filter = ""
    if timing == "morning":
        time_filter = " AND (DepartureTime LIKE '0%' OR DepartureTime LIKE '1%')"
    elif timing == "afternoon":
        time_filter = " AND DepartureTime LIKE '1[2-6]%'"
    elif timing == "evening":
        time_filter = " AND (DepartureTime LIKE '1[7-9]%' OR DepartureTime LIKE '20%')"
    elif timing == "night":
        time_filter = " AND DepartureTime LIKE '2[1-3]%'"

    rows = db.execute(text(f"""
        SELECT FlightId, FlightName, Source, Destination, BasePrice, DepartureTime
        FROM flights WHERE Destination=:dest {time_filter}
        ORDER BY BasePrice ASC LIMIT 3
    """), {"dest": destination}).fetchall()

    if not rows:
        rows = db.execute(text("""
            SELECT FlightId, FlightName, Source, Destination, BasePrice, DepartureTime
            FROM flights WHERE Destination=:dest ORDER BY BasePrice ASC LIMIT 3
        """), {"dest": destination}).fetchall()

    return [dict(r._mapping) for r in rows]


def get_best_bundle(db, city, budget, rating, timing):
    hotels = get_hotels(db, city, budget / 2, rating)
    flights = get_flights(db, city, timing)
    if not hotels or not flights:
        return None
    best, best_score = None, float("inf")
    for f in flights:
        for h in hotels:
            total = float(f["BasePrice"]) + float(h["BasePrice"])
            score = total + max(0, total - budget) * 2
            if score < best_score:
                best_score, best = score, (f, h, total)
    if best:
        f, h, total = best
        return {
            "flight": f, "hotel": h,
            "totalCost": round(total, 2),
            "withinBudget": total <= budget,
            "confidence": round(1 / (1 + best_score / 10000), 2),
        }
    return None


# ── session helpers ───────────────────────────

def _enc(city="", dates="", rating="", timing="") -> str:
    return f"{city}|{dates}|{rating}|{timing}"


def _dec(raw: str):
    p = (raw or "").split("|")
    city   = p[0] if len(p) > 0 else ""
    dates  = p[1] if len(p) > 1 else ""
    rating = int(p[2]) if len(p) > 2 and p[2].isdigit() else None
    timing = p[3] if len(p) > 3 and p[3] else None
    return city, dates, rating, timing


def _save(db, uid, intent, dest_enc, question, budget=None):
    db.execute(text("""
        INSERT INTO chatsession (UserId, LastIntent, LastDestination, LastQuestion, LastBudget)
        VALUES (:uid, :i, :d, :q, :b)
        ON DUPLICATE KEY UPDATE
            LastIntent=:i, LastDestination=:d, LastQuestion=:q, LastBudget=:b
    """), {"uid": uid, "i": intent, "d": dest_enc, "q": question, "b": budget})


def _fmt_hotels(hotels) -> str:
    return "\n".join(
        f"  🏨 {h['HotelName']} — ₹{float(h['BasePrice']):,.0f}/night"
        f" · ⭐{h['Rating']} · {h['AvailableRooms']} rooms left"
        for h in hotels
    ) or "No hotels found."


def _fmt_flights(flights) -> str:
    return "\n".join(
        f"  ✈️ {f['FlightName']} · {f['Source']}→{f['Destination']}"
        f" · ₹{float(f['BasePrice']):,.0f} · {f['DepartureTime']}"
        for f in flights
    ) or "No flights found."


def _r(reply, intent, state, model="State Machine", **kw):
    return {
        "reply": reply,
        "intentDetected": intent,
        "confidence": 1.0,
        "conversationState": state,
        "modelType": model,
        "fallbackTriggered": False,
        "recommendations": kw.get("recommendations"),
        "bundle": kw.get("bundle"),
        "pricing": kw.get("pricing"),
    }


# ══════════════════════════════════════════════
#  MAIN HANDLER
# ══════════════════════════════════════════════

def handle_chat(data):
    db  = SessionLocal()
    msg = data.message.lower().strip()
    uid = data.userId

    # Reset session when user starts a new request
    if any(word in msg for word in ["trip", "travel", "flight", "hotel", "bundle"]):
        db.execute(text("DELETE FROM chatsession WHERE UserId=:uid"), {"uid": uid})
        db.commit()

    d_city   = detect_city(msg)
    d_budget = extract_budget(msg)
    d_date   = extract_date(msg)
    d_rating = extract_rating(msg)
    d_timing = extract_timing(msg)

    row = db.execute(text("""
        SELECT LastIntent, LastDestination, LastQuestion, LastBudget
        FROM chatsession WHERE UserId=:uid
    """), {"uid": uid}).fetchone()

    last_intent = row[0] if row else None
    last_city, last_dates, last_rating, last_timing = _dec(row[1] if row else "")
    last_q      = row[2] if row else None
    last_budget = float(row[3]) if (row and row[3]) else None

    # Safety: reset invalid states
    valid_states = [
        None, "ask_city", "ask_dates", "ask_budget",
        "ask_hotel_city", "ask_flight_city",
        "ask_hotel_rating", "ask_flight_timing",
        "await_confirmation", "completed", "ask_intent"
    ]
    if last_q not in valid_states:
        last_q = None

    def done(reply, intent, state, **kw):
        db.commit()
        db.close()
        return _r(reply, intent, state, **kw)

    # ══════════════════════════════════════════
    #  STATE MACHINE
    # ══════════════════════════════════════════

    # ── ask_city (travel/bundle flow) ──
    if last_q == "ask_city":
        city = d_city or last_city
        if city:
            _save(db, uid, last_intent, _enc(city, last_dates, str(last_rating or ""), last_timing or ""), "ask_dates", last_budget)
            return done(f"📅 {city} noted! When are you travelling? (e.g. 15 Apr, next weekend)", last_intent, "ask_dates")
        return done("I couldn't find that city. Try: Goa, Mumbai, Delhi, Chennai, Bangalore.", last_intent, "ask_city")

    # ── ask_dates (travel/bundle flow) ──
    if last_q == "ask_dates":
        dates = d_date or (msg if len(msg) < 25 else None)
        if dates:
            _save(db, uid, last_intent, _enc(last_city, dates, str(last_rating or ""), last_timing or ""), "ask_budget", last_budget)
            return done(f"📅 {dates} — got it! What's your total budget for {last_city}? (e.g. 10000, 15000)", last_intent, "ask_budget")
        return done("Please share travel dates, e.g. '15 Apr' or 'next weekend'.", last_intent, "ask_dates")

    # ── ask_budget ──
    if last_q == "ask_budget":
        budget = d_budget
        if budget:
            _save(db, uid, last_intent, _enc(last_city, last_dates, str(last_rating or ""), last_timing or ""), "ask_hotel_rating", budget)
            return done(f"💰 ₹{budget:,.0f} noted!\n\n⭐ Hotel star rating preference? (3, 4, 5 — or 'any')", last_intent, "ask_hotel_rating")
        return done("Please share a budget amount, e.g. '10000' or '15000'.", last_intent, "ask_budget")

    # ── ask_hotel_city (hotel flow only) ──
    if last_q == "ask_hotel_city":
        city = d_city
        if city:
            _save(db, uid, "hotel", _enc(city), "ask_budget")
            return done(f"🏨 {city}! What's your budget per night? (e.g. 3000, 5000)", "hotel", "ask_budget")
        return done("Which city? Try: Goa, Chennai, Mumbai, Delhi, Bangalore.", "hotel", "ask_hotel_city")

    # ── ask_flight_city (flight flow only) ──
    if last_q == "ask_flight_city":
        city = d_city
        if city:
            _save(db, uid, "flight", _enc(city), "ask_flight_timing")
            return done(f"✈️ {city}! Preferred departure time?\n(Morning / Afternoon / Evening / Night / any)", "flight", "ask_flight_timing")
        return done("Which city? Try: Goa, Chennai, Mumbai, Delhi, Bangalore.", "flight", "ask_flight_city")

    # ── ask_hotel_rating ──
    if last_q == "ask_hotel_rating":
        rating = d_rating if d_rating else (None if "any" in msg else last_rating)
        budget = last_budget or 10000
        stars = f"{rating}★" if rating else "any"

        if last_intent == "hotel":
            # HOTEL flow: show hotels, done
            hotels = get_hotels(db, last_city, budget / 2, rating)
            _save(db, uid, last_intent, _enc(last_city, "", str(rating or ""), ""), "completed", budget)
            return done(
                f"⭐ Rating: {stars}\n\n🏨 Best hotels in {last_city}:\n\n" + _fmt_hotels(hotels),
                last_intent, "completed",
                recommendations={"hotel": hotels, "flights": None}
            )
        else:
            # TRAVEL/BUNDLE flow: ask flight timing next
            _save(db, uid, last_intent, _enc(last_city, last_dates, str(rating or ""), last_timing or ""), "ask_flight_timing", budget)
            return done(
                f"⭐ Hotel rating: {stars}\n\n"
                f"🕐 Preferred flight departure time?\n"
                f"(Morning / Afternoon / Evening / Night / any)",
                last_intent, "ask_flight_timing"
            )

    # ── ask_flight_timing ──
    if last_q == "ask_flight_timing":
        if not d_timing and "any" not in msg:
            return done(
                "🕐 Please choose: Morning / Afternoon / Evening / Night (or type 'any')",
                last_intent, "ask_flight_timing"
            )
        timing = d_timing or "any"
        flights = get_flights(db, last_city, timing)

        if last_intent == "flight":
            # FLIGHT flow: show flights, done
            _save(db, uid, last_intent, _enc(last_city, "", "", timing), "completed", last_budget)
            return done(
                f"✈️ Best flights to {last_city} ({timing}):\n\n" + _fmt_flights(flights),
                last_intent, "completed",
                recommendations={"hotel": None, "flights": flights}
            )
        else:
            # TRAVEL/BUNDLE flow: show flights + ask confirmation
            _save(db, uid, last_intent, _enc(last_city, last_dates, str(last_rating or ""), timing), "await_confirmation", last_budget)
            return done(
                f"✈️ Best flights to {last_city} ({timing}):\n\n"
                + _fmt_flights(flights)
                + "\n\n👉 Type 'confirm' to book the bundle or 'edit' to change.",
                last_intent, "await_confirmation",
                recommendations={"hotel": None, "flights": flights}
            )

    # ── await_confirmation (travel/bundle flow) ──
    if last_q == "await_confirmation":
        if any(w in msg for w in ["confirm", "yes", "book", "ok"]):
            city   = last_city or "Goa"
            budget = last_budget or 10000
            timing = last_timing or "any"
            bundle = get_best_bundle(db, city, budget, last_rating, timing)

            if not bundle:
                return done("⚠️ No complete package found. Try different options.", last_intent, "completed")

            f = bundle["flight"]
            h = bundle["hotel"]
            reply = (
                f"🎉 Booking Confirmed!\n\n"
                f"✈️ {f['FlightName']} · {f['DepartureTime']} · ₹{float(f['BasePrice']):,.0f}\n"
                f"🏨 {h['HotelName']} · ⭐{h['Rating']} · ₹{float(h['BasePrice']):,.0f}\n\n"
                f"💰 Total: ₹{bundle['totalCost']:,.0f}"
            )
            return done(reply, last_intent, "completed", bundle=bundle)

        if any(w in msg for w in ["edit", "change", "no"]):
            return done("No problem 👍 Let's start again. Which city?", last_intent, "ask_city")

        return done("👉 Type 'confirm' to book or 'edit' to change.", last_intent, "await_confirmation")

    # ══════════════════════════════════════════
    #  INTENT CLASSIFIER (fresh conversation)
    # ══════════════════════════════════════════

    intent, confidence = predict_intent_with_confidence(msg)

    if confidence < INTENT_CONFIDENCE_THRESHOLD:
        db.close()
        return _r(
            "I'm not sure what you mean. Are you looking for travel, hotel, flight, or a bundle?",
            intent, "unknown", model="ML Intent Classifier"
        )

    # ── Greeting ──
    if intent == "greeting":
        _save(db, uid, "greeting", "", "ask_intent")
        return done(
            "👋 Hi! I'm your TravelAI assistant.\nAre you looking for a travel package, hotel, flight, or bundle?",
            intent, "ask_intent"
        )

    # ── Flight: city → timing → flights ──
    if intent == "flight":
        if d_city:
            _save(db, uid, "flight", _enc(d_city), "ask_flight_timing")
            return done(
                f"✈️ Flying to {d_city}! Preferred departure time?\n(Morning / Afternoon / Evening / Night / any)",
                "flight", "ask_flight_timing"
            )
        _save(db, uid, "flight", _enc(), "ask_flight_city")
        return done("✈️ Where are you flying to? (Goa, Mumbai, Delhi, Chennai, Bangalore)", "flight", "ask_flight_city")

    # ── Hotel: city → budget → rating → hotels ──
    if intent == "hotel":
        if d_city and d_budget:
            _save(db, uid, "hotel", _enc(d_city), "ask_hotel_rating", d_budget)
            return done(
                f"🏨 {d_city} · ₹{d_budget:,.0f}/night!\n\n⭐ Star rating preference? (3, 4, 5 — or 'any')",
                "hotel", "ask_hotel_rating"
            )
        if d_city:
            _save(db, uid, "hotel", _enc(d_city), "ask_budget")
            return done(f"🏨 {d_city} noted! What's your budget per night?", "hotel", "ask_budget")
        _save(db, uid, "hotel", _enc(), "ask_hotel_city")
        return done("🏨 Which city do you need a hotel in?", "hotel", "ask_hotel_city")

    # ── Travel/Bundle: city → dates → budget → rating → timing → bundle ──
    if intent in ("travel", "trip", "bundle"):
        if d_city:
            _save(db, uid, "travel", _enc(d_city), "ask_dates")
            return done(
                f"✈️ {d_city}! When are you planning to travel? (e.g. 15 Apr, next weekend)",
                "travel", "ask_dates"
            )
        _save(db, uid, "travel", _enc(), "ask_city")
        return done(
            "✈️ Let's plan your trip! Which city?\n(Goa, Mumbai, Delhi, Chennai, Bangalore)",
            "travel", "ask_city"
        )

    # ── Shopping ──
    if intent == "shopping":
        _save(db, uid, "shopping", "", "ask_shopping_details")
        return done("🛍️ What are you looking to buy?", intent, "ask_shopping_details")

    # ── Fallback ──
    db.close()
    return _r(
        "I can help with travel packages, hotels, flights, or bundles. What are you looking for?",
        intent, "unknown", model="ML Intent Classifier"
    )
