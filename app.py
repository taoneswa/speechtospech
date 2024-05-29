import streamlit as st
import speech_recognition as sr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pyaudio
import wave

# Load the model and tokenizer
model_name = 't5-base'
model_path = 't5_shona_english_model.pth'
tokenizer_path = '/t5_tokenizer/'

# Initialize model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.load_state_dict(torch.load(model_path))
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model.eval()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to record audio
def record_audio():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        st.write("Recognizing...")
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Google Speech Recognition could not understand the audio")
            return ""
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
            return ""

# Function to translate text
def translate_text(text, source_lang='en', target_lang='shona'):
    input_text = f"translate {source_lang} to {target_lang}: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Streamlit interface
st.title("Live Speech Translator")

option = st.selectbox("Select translation direction:", ("English to Shona", "Shona to English"))

if st.button("Start Recording"):
    recognized_text = record_audio()
    if recognized_text:
        if option == "English to Shona":
            translated_text = translate_text(recognized_text, source_lang='en', target_lang='shona')
        else:
            translated_text = translate_text(recognized_text, source_lang='shona', target_lang='en')
        st.write(f"Translated Text: {translated_text}")
        # You can add TTS here if needed

