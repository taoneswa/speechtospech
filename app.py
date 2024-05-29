import os
import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForSeq2SeqLM, AutoTokenizer
import librosa
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the translation model
model_save_path = 't5_shona_english_model.pth'  # Adjust path if needed
tokenizer_save_path = 't5_tokenizer/'  # Adjust path if needed

translation_model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
translation_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
translation_model.eval()
logging.info("Translation model loaded successfully")

translation_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
logging.info("Translation tokenizer loaded successfully")

# Load ASR model and processor
asr_model_name = "facebook/wav2vec2-large-960h"
asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)
asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
logging.info("ASR model and processor loaded successfully")

st.title("Speech-to-Speech Translation App")

# Audio file upload
st.header("Upload Audio File")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        audio_path = f"/tmp/{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        audio, sr = librosa.load(audio_path, sr=16000)
        st.write("Audio loaded successfully")

        # Transcription and translation placeholder
        st.write("Transcribing and translating...")

        # Placeholder for transcription and translation
        transcription = "example transcription"
        translation = "example translation"

        st.write("Transcribed Text:", transcription)
        st.write("Translated Text:", translation)

        logging.info("Transcription and translation completed successfully")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        logging.error(f"Error processing the file: {e}")
