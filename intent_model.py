import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# ===============================
# EXPANDED TRAINING DATA
# ===============================
texts = [

    # ── TRAVEL (50+ examples) ───────────────────────────────
    "i want to travel",
    "plan my trip",
    "go to goa",
    "book flight",
    "travel to mumbai",
    "i want to go bangalore",
    "plan trip to goa",
    "plan a trip to chennai",
    "planning a vacation",
    "i want to visit delhi",
    "trip to bangalore",
    "want to go to mumbai",
    "book a trip",
    "going to goa next week",
    "flight to delhi",
    "travel to bangalore",
    "visit chennai",
    "i want to fly to goa",
    "need flight tickets",
    "book flight to mumbai",
    "plan vacation to goa",
    "want to travel to chennai",
    "holiday trip to bangalore",
    "weekend trip to goa",
    "family trip to delhi",
    "travel plan",
    "i am going to goa",
    "trip planning",
    "book air ticket",
    "fly to bangalore",
    "i want to go on a trip",
    "looking for flights",
    "need to travel",
    "going on vacation",
    "plan holiday",
    "trip to mumbai",
    "travel to goa",
    "i want to visit mumbai",
    "book travel",
    "need air ticket to goa",
    "want to fly",
    "going to chennai",
    "vacation to bangalore",
    "road trip",
    "travel itinerary",
    "cheap flights to goa",
    "one way ticket to delhi",
    "round trip to mumbai",
    "how to go to goa",
    "best way to travel to chennai",
    "i need to go to bangalore",
    "suggest a trip",
    "travel package",
    "tour package",
    "plan my vacation",

    # ── HOTEL (40+ examples) ────────────────────────────────
    "book hotel",
    "find hotel",
    "need stay",
    "hotel in chennai",
    "cheap hotel",
    "hotel booking",
    "book a room",
    "need a hotel in goa",
    "hotel near me",
    "find accommodation",
    "looking for hotel",
    "budget hotel",
    "luxury hotel",
    "hotel in mumbai",
    "resort in goa",
    "5 star hotel",
    "hotel for 2 nights",
    "stay in bangalore",
    "accommodation in delhi",
    "need a place to stay",
    "book resort",
    "hotel with pool",
    "hotel near airport",
    "find me a hotel",
    "cheapest hotel in goa",
    "best hotels in chennai",
    "hotel room booking",
    "need lodging",
    "rent a room",
    "hostel in bangalore",
    "hotel under 3000",
    "affordable stay",
    "hotel for tonight",
    "hotel in delhi",
    "guest house",
    "service apartment",
    "hotel deals",
    "hotel near beach",
    "hotel with breakfast",
    "boutique hotel",

    # ── SHOPPING (30+ examples) ─────────────────────────────
    "buy shoes",
    "purchase mobile",
    "order product",
    "buy headphones",
    "shopping",
    "buy something",
    "order online",
    "purchase laptop",
    "buy travel bag",
    "shop for clothes",
    "buy sunscreen",
    "order food",
    "purchase camera",
    "buy travel insurance",
    "shopping for trip",
    "need to buy",
    "looking to purchase",
    "buy tickets",
    "buy souvenirs",
    "purchase accessories",
    "order luggage",
    "buy travel pillow",
    "purchase backpack",
    "shop online",
    "buy sunglasses",
    "purchase earphones",
    "buy medicines",
    "order products",
    "buy items",
    "purchase things",

    # ── GREETING (20+ examples) ─────────────────────────────
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening",
    "good afternoon",
    "good night",
    "hi there",
    "hey there",
    "hello there",
    "howdy",
    "what's up",
    "greetings",
    "namaste",
    "vanakkam",
    "how are you",
    "nice to meet you",
    "start",
    "begin",
    "help",
]

labels = [
    # travel (55)
    "travel","travel","travel","travel","travel","travel",
    "travel","travel","travel","travel","travel","travel",
    "travel","travel","travel","travel","travel","travel",
    "travel","travel","travel","travel","travel","travel",
    "travel","travel","travel","travel","travel","travel",
    "travel","travel","travel","travel","travel","travel",
    "travel","travel","travel","travel","travel","travel",
    "travel","travel","travel","travel","travel","travel",
    "travel","travel","travel","travel","travel","travel",
    "travel",                                              # ← this was missing
    # hotel (40)
    "hotel","hotel","hotel","hotel","hotel","hotel","hotel","hotel",
    "hotel","hotel","hotel","hotel","hotel","hotel","hotel","hotel",
    "hotel","hotel","hotel","hotel","hotel","hotel","hotel","hotel",
    "hotel","hotel","hotel","hotel","hotel","hotel","hotel","hotel",
    "hotel","hotel","hotel","hotel","hotel","hotel","hotel","hotel",
    # shopping (30)
    "shopping","shopping","shopping","shopping","shopping","shopping",
    "shopping","shopping","shopping","shopping","shopping","shopping",
    "shopping","shopping","shopping","shopping","shopping","shopping",
    "shopping","shopping","shopping","shopping","shopping","shopping",
    "shopping","shopping","shopping","shopping","shopping","shopping",
    # greeting (20)
    "greeting","greeting","greeting","greeting","greeting","greeting",
    "greeting","greeting","greeting","greeting","greeting","greeting",
    "greeting","greeting","greeting","greeting","greeting","greeting",
    "greeting","greeting",
]

# ===============================
# TRAIN MODEL
# ===============================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # bigrams — captures "plan trip", "book flight" etc.
    min_df=1,
    analyzer='word'
)
X = vectorizer.fit_transform(texts)

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver='lbfgs'
)
model.fit(X, labels)

# ===============================
# CROSS VALIDATION CHECK
# ===============================
scores = cross_val_score(model, X, labels, cv=3, scoring='accuracy')
print(f"✅ Cross-val accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

# ===============================
# QUICK SMOKE TEST
# ===============================
test_phrases = [
    "plan trip to goa",
    "hotel in chennai",
    "buy laptop",
    "hello",
    "i want to visit mumbai",
    "book a hotel",
    "trip to bangalore under 5000",
]
print("\n📋 Smoke test:")
for phrase in test_phrases:
    X_test = vectorizer.transform([phrase])
    pred = model.predict(X_test)[0]
    proba = model.predict_proba(X_test)[0]
    conf = float(proba[list(model.classes_).index(pred)])
    print(f"  '{phrase}' → {pred} ({conf:.2f})")

# ===============================
# SAVE
# ===============================
joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "intent_vectorizer.pkl")
print("\n✅ Intent model trained and saved!")
