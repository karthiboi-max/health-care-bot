import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
from transformers import pipeline

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if device == "cuda" else -1)

qa_pipeline = load_model()

def get_answer(question, context):
    response = qa_pipeline({"question": question, "context": context})
    return response.get("answer", "I am not sure about that. Please consult a healthcare professional.")

# Streamlit UI
st.title("AI Healthcare Chatbot")
st.write("Ask any health-related question, and the AI will provide answers.")

user_question = st.text_input("Enter your health-related question:")
if user_question:
    user_tokens = word_tokenize(user_question.lower())
    filtered_tokens = [word for word in user_tokens if word not in stopwords.words("english")]
    
    if "appointment" in filtered_tokens:
        st.write("Would you like to schedule an appointment?")
    elif any(word in filtered_tokens for word in ["medication", "symptom", "prescription", "treatment"]):
        st.write("It is a better option to consult a nearby doctor.")
    elif any(word in filtered_tokens for word in ["cold", "fever", "headache", "flu", "chills", "sore throat", "runny nose"]):
        st.write("Common cold and fever are usually viral. Stay hydrated, get rest, and if symptoms persist, consult a doctor.")
    else:
        answer = get_answer(user_question, medical_context)
        st.write("**AI's Answer:**", answer)

# Adding an option for users to schedule an appointment
if st.button("Schedule an Appointment"):
    st.write("Feature coming soon: You will be able to book an appointment directly!")
