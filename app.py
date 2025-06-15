# app.py

import streamlit as st
from utils import extract_text_from_pdf
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ Page config
st.set_page_config(page_title="Flashcard Generator", page_icon="🧠")
st.title("📚 Flashcard Generator (TinyLlama - Offline)")

# ✅ Load TinyLlama locally
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ✅ Generate flashcards using prompt formatting
def generate_flashcards(text, num=10):
    try:
        # Clean and shorten the content
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        content = "\n".join(lines[:15])

        # Strong and structured prompt
        prompt = f"""
You are a helpful AI assistant.

Create {num} flashcards based on the following notes. Each flashcard should be in this format:

Q: [Question]
A: [Answer]

Notes:
{content}

Flashcards:
Q:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean and extract flashcard block
        return result[result.find("Q:"):].strip() if "Q:" in result else result.strip()

    except Exception as e:
        return f"❌ Flashcard generation error: {str(e)}"

# ✅ Streamlit UI
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF text extracted successfully.")

    if st.button("Generate Flashcards"):
        with st.spinner("⏳ Generating flashcards using TinyLlama..."):
            flashcards = generate_flashcards(text)
            st.subheader("Flashcards:")
            st.text_area("Flashcard Output", flashcards, height=300)
