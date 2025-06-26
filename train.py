# Step 1: Install required libraries (only needed once in colab/local)
# !pip install transformers datasets scikit-learn

# Step 2: Imports
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import os
os.environ["WANDB_DISABLED"] = "true"


# Step 3: Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Step 4: Load CSV and fix label mapping
df = pd.read_csv("/content/emotions.csv")

# Normalize labels to match the model
label_map = {
    "sad": "sadness",
    "joy":"joy",
    "love": "love",
    "anger":"anger",
    "fear":"fear",
    
    "surprise":"surprise"
}
df['label'] = df['label'].map(label_map).fillna(df['label'])

# Step 5: Encode labels
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])

# Save mapping for later use
label2id = {label: idx for idx, label in enumerate(le.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

# Step 6: Convert to Hugging Face Dataset format
# Rename 'label_id' to 'labels' for the Trainer to automatically use it
dataset = Dataset.from_pandas(df[['sentence', 'label_id']].rename(columns={'label_id': 'labels'}))

# Step 7: Split dataset
dataset = dataset.train_test_split(test_size=0.2)
train_ds = dataset['train']
test_ds = dataset['test']

# Step 8: Load tokenizer and model
model_name = "nateraw/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True)

# Map tokenization function
train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# Step 9: Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)

# Step 10: Training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./emotion_model",
    eval_strategy="epoch", # Changed from evaluation_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    run_name="laksh_v2",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


# Step 11: Metrics
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids # This is correctly referencing 'label_ids' from the dataset
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Step 12: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Step 13: Train
trainer.train()

# Step 14: Save final model
trainer.save_model("./emotion_model/fine_tuned")
tokenizer.save_pretrained("./emotion_model/fine_tuned")
