
import re
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# Step 1: Load saved model and vectorizer

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Step 2: Create FastAPI application

app = FastAPI(
    title="NLP SMS Spam Detector",
    description="API to classify SMS messages as spam or ham",
    version="1.0"
)

# Step 3: Define JSON input format

class SMSRequest(BaseModel):
    message: str


# Step 4: Clean input message

def clean_text(message):
    message = message.lower()
    message = re.sub(r"http\S+|www\S+", " ", message)
    message = re.sub(r"[^a-z\s]", " ", message)
    message = re.sub(r"\s+", " ", message)
    return message.strip()


# Step 5: Create prediction endpoint

@app.post("/predict")
def predict_sms(request: SMSRequest):

    message = clean_text(request.message)

    if not message:
        return {
            "error": "Please enter a non-empty SMS message."
        }

    # Convert message into TF-IDF features

    message_tfidf = vectorizer.transform([message])

    # Predict class

    prediction = model.predict(message_tfidf)[0]

    # Calculate prediction confidence

    probabilities = model.predict_proba(message_tfidf)[0]

    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
