
import os
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------------
# Step 1: Set device
# -----------------------------------

device = torch.device("cpu")

print("Device:", device, flush=True)

# -----------------------------------
# Step 2: Load dataset
# -----------------------------------

print("\nLoading dataset...", flush=True)

df = pd.read_csv("customer_support_tickets.csv")

# Use feedback column if available

if "feedback" in df.columns:

    df["feedback"] = df["feedback"].fillna("").astype(str)

else:

    df["Ticket Subject"] = df["Ticket Subject"].fillna("").astype(str)

    df["Ticket Description"] = df["Ticket Description"].fillna("").astype(str)

    df["feedback"] = (
        df["Ticket Subject"] + " " + df["Ticket Description"]
    )

df["Ticket Type"] = df["Ticket Type"].fillna("").astype(str)

# Remove empty records

df = df[
    (df["feedback"].str.strip() != "") &
    (df["Ticket Type"].str.strip() != "")
].copy()

# -----------------------------------
# Step 3: Reduce dataset size
# -----------------------------------

print("\nReducing dataset size...", flush=True)

# Select 200 records from each category

df = df.groupby(
    "Ticket Type",
    group_keys=False
).head(200)

df = df.reset_index(drop=True)

print("Reduced Dataset Shape:", df.shape)

print("\nCategory Distribution:")
print(df["Ticket Type"].value_counts())

# -----------------------------------
# Step 4: Prepare text and labels
# -----------------------------------

texts = df["feedback"].tolist()

labels = df["Ticket Type"].tolist()

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)

num_labels = len(label_encoder.classes_)

print("\nCategories:", list(label_encoder.classes_))

# -----------------------------------
# Step 5: Split dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# -----------------------------------
# Step 6: Load BERT tokenizer
# -----------------------------------

print("\nLoading BERT tokenizer...", flush=True)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

# -----------------------------------
# Step 7: Create dataset class
# -----------------------------------

class FeedbackDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        text = str(self.texts[index])

        encoding = self.tokenizer(
            text,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),

            "attention_mask": encoding["attention_mask"].squeeze(0),

            "labels": torch.tensor(
                int(self.labels[index]),
                dtype=torch.long
            )
        }

        return item

# -----------------------------------
# Step 8: Create DataLoaders
# -----------------------------------

train_dataset = FeedbackDataset(
    X_train,
    y_train,
    tokenizer
)

test_dataset = FeedbackDataset(
    X_test,
    y_test,
    tokenizer
)

batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("\nTraining batches:", len(train_loader))
print("Testing batches:", len(test_loader))

# -----------------------------------
# Step 9: Load BERT model
# -----------------------------------

print("\nLoading BERT model...", flush=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)

model.to(device)

# -----------------------------------
# Step 10: Set optimizer
# -----------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

epochs = 1

# -----------------------------------
# Step 11: Train BERT model
# -----------------------------------

print("\nStarting BERT training...", flush=True)

for epoch in range(epochs):

    model.train()

    total_loss = 0

    print(
        "\nEpoch",
        epoch + 1,
        "of",
        epochs,
        "started",
        flush=True
    )

    for batch_index, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels_batch = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # Display progress every 5 batches

        if (batch_index + 1) % 5 == 0:

            print(
                "Epoch",
                epoch + 1,
                "| Batch",
                batch_index + 1,
                "of",
                len(train_loader),
                "| Loss:",
                round(loss.item(), 4),
                flush=True
            )

    average_loss = total_loss / len(train_loader)

    print(
        "\nEpoch completed!",
        "Average Loss:",
        round(average_loss, 4),
        flush=True
    )

print("\nTraining completed!", flush=True)

# -----------------------------------
# Step 12: Evaluate model
# -----------------------------------

print("\nEvaluating model...", flush=True)

model.eval()

all_predictions = []

all_actual_labels = []

with torch.no_grad():

    for batch_index, batch in enumerate(test_loader):

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels_batch = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_actual_labels.extend(
            labels_batch.cpu().tolist()
        )

        if (batch_index + 1) % 5 == 0:

            print(
                "Evaluated",
                batch_index + 1,
                "of",
                len(test_loader),
                "batches",
                flush=True
            )

# -----------------------------------
# Step 13: Calculate metrics
# -----------------------------------

accuracy = accuracy_score(
    all_actual_labels,
    all_predictions
)

precision = precision_score(
    all_actual_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_actual_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_actual_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy, 4))

print("Precision:", round(precision, 4))

print("Recall:", round(recall, 4))

print("F1-Score:", round(f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        all_actual_labels,
        all_predictions,
        labels=list(range(num_labels)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# -----------------------------------
# Step 14: Create confusion matrix
# -----------------------------------

cm = confusion_matrix(
    all_actual_labels,
    all_predictions,
    labels=list(range(num_labels))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

display.plot(
    xticks_rotation=45,
    values_format="d"
)

plt.title("BERT Customer Feedback Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.close()

print("\nConfusion matrix saved successfully!")

# -----------------------------------
# Step 15: Save trained model
# -----------------------------------

save_path = "saved_bert_feedback_model"

os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)

tokenizer.save_pretrained(save_path)

with open(
    os.path.join(save_path, "labels.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        label_encoder.classes_.tolist(),
        file
    )

print("\nModel saved successfully!")

# -----------------------------------
# Step 16: Predict new feedback
# -----------------------------------

print("\n========== NEW FEEDBACK PREDICTION ==========")

new_feedback = input(
    "\nEnter customer feedback: "
)

model.eval()

encoding = tokenizer(
    new_feedback,
    max_length=32,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():

    outputs = model(
        input_ids=encoding["input_ids"].to(device),
        attention_mask=encoding["attention_mask"].to(device)
    )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][predicted_class].item()

predicted_category = label_encoder.inverse_transform(
    [predicted_class]
)[0]

print("\nPredicted Category:", predicted_category)

print(
    "Model Confidence:",
    round(confidence * 100, 2),
    "%"
)

print("\nProgram completed successfully!")
