import joblib

# Load models (VERY IMPORTANT)
intent_model = joblib.load("intent_model.pkl")
intent_vectorizer = joblib.load("intent_vectorizer.pkl")

def predict_intent_with_confidence(message: str):
    X = intent_vectorizer.transform([message])
    probs = intent_model.predict_proba(X)[0]

    intent_index = probs.argmax()
    confidence = probs[intent_index]
    intent = intent_model.classes_[intent_index]

    return intent, confidence
