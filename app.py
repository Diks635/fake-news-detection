import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Fake News Detector", layout="centered")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "model/distilbert_fake_news",
        local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        
        "model/distilbert_fake_news",
        local_files_only=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("📰 Fake News Detector for Students")
st.write(
    "Paste a news article to check whether it is **REAL**, **FAKE**, "
    "or **UNCERTAIN** using AI."
)

news_text = st.text_area("Paste News Text Here", height=200)

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        inputs = tokenizer(
            news_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

        st.subheader("🔍 Prediction Result")

        if real_prob >= 0.60:
            st.success(f"✅ REAL NEWS\n\nConfidence: **{real_prob:.2f}**")
        elif fake_prob >= 0.60:
            st.error(f"❌ FAKE NEWS\n\nConfidence: **{fake_prob:.2f}**")
        else:
            st.warning(
                "⚠️ UNCERTAIN NEWS\n\n"
                f"Real: {real_prob:.2f} | Fake: {fake_prob:.2f}\n\n"
                "Manual fact-checking recommended."
            )

        st.caption(
            "ℹ️ This AI analyzes language patterns, not real-time facts. "
            "Always verify with trusted sources."
        )
