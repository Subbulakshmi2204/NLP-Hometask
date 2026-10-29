
# Step 1: Import Libraries

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW


# Step 2: Select CPU or GPU

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# Step 3: Load AG News Dataset

# Step 3: Load AG News Dataset

import pandas as pd

def load_dataset(file_path):

    df = pd.read_csv(file_path, header=None)

    # Check whether the first row contains column names
    first_value = str(df.iloc[0, 0]).strip().lower()

    if first_value in ["class index", "label"]:
        df = pd.read_csv(file_path)

        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace("_", " ")
        )

        df = df.rename(columns={
            "class index": "label",
            "title": "title",
            "description": "description"
        })

    else:
        df.columns = [
            "label",
            "title",
            "description"
        ]

    return df


train_df = load_dataset("agtrain.csv")
test_df = load_dataset("agtest.csv")

print("Training Dataset:", train_df.shape)
print("Testing Dataset:", test_df.shape)

print("\nFirst Five Records:")
print(train_df.head())

print("\nCategory Distribution:")
print(train_df["label"].value_counts().sort_index())




# Step 4: Explore Four Categories

categories = [
    "World",
    "Sports",
    "Business",
    "Technology"
]

print("\nCategory Distribution:")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

train_df["category"] = (
    train_df["label"].map({
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Technology"
    })
)

train_df["category"].value_counts().plot(
    kind="bar",
    rot=0
)

plt.title("AG News Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()


# Step 5: Prepare Text and Labels

train_df["text"] = (
    train_df["title"].fillna("").astype(str)
    + " "
    + train_df["description"].fillna("").astype(str)
)
test_df["label"] = test_df["label"].astype(int) - 1
test_df["text"] = (
    test_df["title"].fillna("").astype(str)
    + " "
    + test_df["description"].fillna("").astype(str)
)

# Convert labels from 1-4 to 0-3
train_df["label"] = train_df["label"].astype(int) - 1
test_df["label"] = test_df["label"].astype(int) - 1

# Use 20,000 samples for a faster first experiment.
# Change to None to use the full training dataset.

TRAIN_LIMIT = 20000

if TRAIN_LIMIT is not None:
    train_df, _ = train_test_split(
        train_df,
        train_size=min(TRAIN_LIMIT, len(train_df)),
        random_state=42,
        stratify=train_df["label"]
    )

# Split training data into training and validation sets
train_texts, val_texts, train_labels, val_labels = (
    train_test_split(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"]
    )
)


# Step 6: Load Pretrained Transformer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label={
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Technology"
    },
    label2id={
        "World": 0,
        "Sports": 1,
        "Business": 2,
        "Technology": 3
    }
)

model.to(device)


# Step 7: Create Dataset Class

class NewsDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        }

        return item


# Step 8: Create DataLoaders

train_dataset = NewsDataset(
    train_texts,
    train_labels
)

val_dataset = NewsDataset(
    val_texts,
    val_labels
)

test_dataset = NewsDataset(
    test_df["text"].tolist(),
    test_df["label"].tolist()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8
)


# Step 9: Display Tokenization Example

sample = tokenizer(
    train_texts[0],
    truncation=True,
    padding="max_length",
    max_length=128
)

print("\nSample News Article:")
print(train_texts[0])

print("\nInput IDs:")
print(sample["input_ids"][:20])

print("\nAttention Mask:")
print(sample["attention_mask"][:20])


# Step 10: Fine-Tune the Transformer

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

epochs = 2

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch in train_loader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print("\nEpoch:", epoch + 1)
    print("Training Loss:", round(average_loss, 4))


    # Validate after each epoch

    model.eval()

    val_actual = []
    val_predictions = []

    with torch.no_grad():

        for batch in val_loader:

            labels = batch["labels"].tolist()

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**inputs)

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            ).cpu().tolist()

            val_actual.extend(labels)
            val_predictions.extend(predictions)

    val_accuracy = accuracy_score(
        val_actual,
        val_predictions
    )

    print(
        "Validation Accuracy:",
        round(val_accuracy * 100, 2),
        "%"
    )


# Step 11: Evaluate on the Test Dataset

model.eval()

actual = []
predictions = []

with torch.no_grad():

    for batch in test_loader:

        labels = batch["labels"].tolist()

        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

        outputs = model(**inputs)

        predicted_labels = torch.argmax(
            outputs.logits,
            dim=1
        ).cpu().tolist()

        actual.extend(labels)
        predictions.extend(predicted_labels)


# Step 12: Calculate Evaluation Metrics

accuracy = accuracy_score(
    actual,
    predictions
)

print("\nTest Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        actual,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=categories,
        zero_division=0
    )
)


# Step 13: Display Confusion Matrix

cm = confusion_matrix(
    actual,
    predictions,
    labels=[0, 1, 2, 3]
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=categories,
    yticklabels=categories
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("AG News Confusion Matrix")

plt.tight_layout()
plt.show()


# Step 14: Predict a New News Article

def predict_news(news_text):

    model.eval()

    inputs = tokenizer(
        news_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    category = categories[predicted_class.item()]

    confidence_score = confidence.item() * 100

    print("\nNews Article:", news_text)
    print("Predicted Category:", category)
    print(
        "Confidence:",
        round(confidence_score, 2),
        "%"
    )


# Step 15: Accept User Input

news = input(
    "\nEnter a news headline or article: "
)

predict_news(news)


# Step 16: Save the Fine-Tuned Model

model.save_pretrained("saved_news_model")
tokenizer.save_pretrained("saved_news_model")

print("\nModel saved successfully.")
