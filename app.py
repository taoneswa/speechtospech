import streamlit as st
import speech_recognition as sr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from gtts import gTTS
import os

# Load the model and tokenizer
model_name = 't5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the trained model state
model_save_path = 't5_shona_english_model.pth'
model.load_state_dict(torch.load(model_save_path))

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Function to translate text
def translate_text(text, src_lang, tgt_lang):
    input_text = f"translate {src_lang} to {tgt_lang}: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# Streamlit UI
st.title("Live Speech Translation")

# Selection for source and target languages
src_lang = st.selectbox("Select source language", ["English", "Shona"])
tgt_lang = "Shona" if src_lang == "English" else "English"

st.write(f"Translating from {src_lang} to {tgt_lang}")

if st.button("Start Recording"):
    try:
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = recognizer.listen(source, timeout=5)

        st.write("Processing...")

        # Recognize speech using Google Speech Recognition
        speech_text = recognizer.recognize_google(audio)
        st.write(f"Recognized text: {speech_text}")

        # Translate the text
        translated_text = translate_text(speech_text, src_lang.lower(), tgt_lang.lower())
        st.write(f"Translated text: {translated_text}")

        # Convert translated text to speech
        tts = gTTS(translated_text, lang='sn' if tgt_lang == 'Shona' else 'en')
        tts.save("translated_audio.mp3")
        os.system("start translated_audio.mp3")  # Adjust this line based on your OS

    except sr.WaitTimeoutError:
        st.write("No speech detected. Please speak into the microphone.")
    except sr.UnknownValueError:
        st.write("Could not understand audio")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")
    except Exception as e:
        st.write(f"An error occurred: {e}")
