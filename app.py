import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import librosa
import numpy as np
import nltk
from nltk.corpus import stopwords
import soundfile as sf
from pydub import AudioSegment

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved translation model and tokenizer
model_save_path = 't5_shona_english_model.pth'
tokenizer_save_path = 't5_tokenizer/'

translation_model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
translation_model.load_state_dict(torch.load(
    model_save_path, map_location=torch.device('cpu')))
translation_model.eval()

translation_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

# Load pre-trained ASR and TTS models
asr_model_name = "facebook/wav2vec2-large-960h"  # Example ASR model
tts_model_name = "tts_transformer"  # Placeholder for TTS model

asr_model = AutoModelForSeq2SeqLM.from_pretrained(asr_model_name)
asr_tokenizer = AutoTokenizer.from_pretrained(asr_model_name)

# Text preprocessing function


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [
        token for token in tokens if token not in stopwords.words('english')]
    return tokens

# Audio preprocessing function


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return audio, sr

# Function to transcribe audio to text


def transcribe_audio(audio, sr):
    inputs = asr_tokenizer(audio, return_tensors="pt", sampling_rate=sr)
    with torch.no_grad():
        logits = asr_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_tokenizer.batch_decode(predicted_ids)[0]
    return transcription

# Function to translate text


def translate_text(text, tokenizer, model):
    preprocessed_text = preprocess_text(text)
    input_text = ' '.join(preprocessed_text)
    encoded_input = tokenizer(
        input_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoded_input['input_ids']

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids)

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Function to synthesize text to audio


def synthesize_speech(text):
    # Placeholder for actual TTS model inference
    # Assume you have a function `synthesize` that converts text to speech
    synthesized_audio = AudioSegment.silent(
        duration=1000)  # Replace with actual TTS synthesis
    return synthesized_audio


st.title("Speech-to-Speech Translation between Shona and English")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('Processing...'):
        audio_path = f"/tmp/{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio, sr = load_audio(audio_path)
        transcription = transcribe_audio(audio, sr)
        st.write("Transcribed Text:", transcription)

        translation = translate_text(
            transcription, translation_tokenizer, translation_model)
        st.write("Translated Text:", translation)

        synthesized_audio = synthesize_speech(translation)

        synthesized_audio_path = f"/tmp/synthesized_{uploaded_file.name}"
        synthesized_audio.export(synthesized_audio_path, format="wav")

        st.audio(synthesized_audio_path, format="audio/wav")
