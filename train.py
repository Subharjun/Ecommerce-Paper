import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration based on Sentilytics AI paper (Section 3.2)
MODEL_NAME = "distilbert-base-uncased"
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 3
WEIGHT_DECAY = 0.01

def prepare_data(csv_path):
    """
    Simulates loading the 300k reviews mentioned in the paper.
    In practice, user should provide their full dataset here.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Mapping sentiments to IDs: Positive (1), Negative (0), Neutral (2)
    # The paper uses 3 classes
    df['label'] = df['Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})
    
    # Split: 70% Train, 15% Val, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["Review"], padding="max_length", truncation=True)

def train_model(train_df, val_df):
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Tokenize
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Model
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    # Training Arguments (Methodology Section 3.2)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1, # 10% warmup as per paper
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=LEARNING_RATE,
        optim="adamw_torch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save optimized model
    model.save_pretrained("./sentilytics_fine_tuned")
    tokenizer.save_pretrained("./sentilytics_fine_tuned")
    print("Training complete. Model saved to ./sentilytics_fine_tuned")

    # Quantization Step (Mentioned in Section 3.2)
    print("Applying Dynamic Quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model.to("cpu"), {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), "./sentilytics_fine_tuned/quantized_model.bin")
    print("Quantized model saved.")

if __name__ == "__main__":
    # In a real scenario, use the full 300k review dataset
    # For now, we use the sample file we created
    try:
        train_df, val_df, test_df = prepare_data("sample_reviews.csv")
        train_model(train_df, val_df)
    except Exception as e:
        print(f"Training setup error: {e}")
        print("Note: Ensure sample_reviews.csv exists and has at least a few rows for demonstration.")
