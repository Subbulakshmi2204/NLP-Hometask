
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Step 1: Load dataset

print("Loading SMS dataset...", flush=True)

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

df = df.dropna()
df = df.drop_duplicates(subset="message")

print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

# Step 2: Clean SMS messages

def clean_text(message):
    message = message.lower()
    message = re.sub(r"http\S+|www\S+", " ", message)
    message = re.sub(r"[^a-z\s]", " ", message)
    message = re.sub(r"\s+", " ", message)
    return message.strip()


df["message"] = df["message"].apply(clean_text)

# Remove empty messages

df = df[df["message"] != ""]

X = df["message"]
y = df["label"]

# Step 3: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Step 4: Convert text into TF-IDF features

print("\nCreating TF-IDF features...", flush=True)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train Logistic Regression

print("Training Logistic Regression...", flush=True)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Training completed!")

# Step 6: Evaluate the model

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="spam",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="spam",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="spam",
    zero_division=0
)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", round(accuracy, 4))
print("Spam Precision:", round(precision, 4))
print("Spam Recall:", round(recall, 4))
print("Spam F1-Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Save model and vectorizer

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved as spam_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")
print("\nTraining and evaluation completed!")
