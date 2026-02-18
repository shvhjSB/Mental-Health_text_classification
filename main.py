
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords only once
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = load_stopwords()

# Load model, tokenizer, label encoder
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "saved_mental_status_bert"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "saved_mental_status_bert"
    )
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

    model.eval()  # IMPORTANT
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model()


# Text cleaning function
def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# Prediction function
def detect_anxiety(text):
    cleaned_text = clean_statement(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200
    )

    with torch.no_grad():  # IMPORTANT
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]


# Streamlit UI
st.title("🧠 Mental Health Status Detection (BERT)")

input_text = st.text_area("Enter your mental state:")

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        predicted_class = detect_anxiety(input_text)
        st.success(f"Predicted Status: **{predicted_class}**")
