from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("model/distilbert_fake_news")

print("✅ Tokenizer re-saved successfully")
