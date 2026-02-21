from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

MODEL_DIR = "model/distilbert_fake_news"

# Load fresh base model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

# Save clean config + weights + tokenizer
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("✅ Clean model + config + tokenizer saved successfully")
