import os
import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForSeq2SeqLM, AutoTokenizer
import librosa
import nltk
from nltk.corpus import stopwords
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, ClientSettings
import numpy as np

# Ensure necessary NLTK data is downloaded
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Load the saved translation model and tokenizer
model_save_path = 't5_shona_english_model.pth'  # Adjust path if needed
tokenizer_save_path = 't5_tokenizer/'  # Adjust path if needed

translation_model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
translation_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
translation_model.eval()

translation_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

# Load pre-trained ASR model and processor
asr_model_name = "facebook/wav2vec2-large-960h"
asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)
asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)

# Text preprocessing function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

# Function to transcribe audio to text
def transcribe_audio(audio, sr):
    inputs = asr_processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = asr_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(predicted_ids)[0]
    return transcription

# Function to translate text
def translate_text(text, tokenizer, model):
    preprocessed_text = preprocess_text(text)
    input_text = ' '.join(preprocessed_text)
    encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoded_input['input_ids']
    
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids)
    
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Function to synthesize text to audio
def synthesize_speech(text):
    # Placeholder for actual TTS model inference
    # Assume you have a function `synthesize` that converts text to speech
    synthesized_audio = AudioSegment.silent(duration=1000)  # Replace with actual TTS synthesis
    return synthesized_audio

# Define a custom audio processor class for webrtc_streamer
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_buffer = []

    def recv(self, frame):
        self.audio_buffer.append(frame.to_ndarray())

    def get_audio(self):
        return self.audio_buffer

st.title("Speech-to-Speech Translation between Shona and English")

# Audio recording
st.header("Record Audio")
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True},
    ),
    audio_processor_factory=AudioProcessor,
)

# Process recorded audio
if webrtc_ctx.state.playing:
    if st.button("Stop Recording"):
        audio_processor = webrtc_ctx.audio_processor
        if audio_processor:
            audio_buffer = audio_processor.get_audio()
            audio_data = np.concatenate(audio_buffer)
            sr = 16000  # Default sample rate for Wav2Vec2
            transcription = transcribe_audio(audio_data, sr)
            st.write("Transcribed Text:", transcription)
            
            translation = translate_text(transcription, translation_tokenizer, translation_model)
            st.write("Translated Text:", translation)
            
            synthesized_audio = synthesize_speech(translation)
            
            synthesized_audio_path = "/tmp/synthesized_recording.wav"
            synthesized_audio.export(synthesized_audio_path, format="wav")
            
            st.audio(synthesized_audio_path, format="audio/wav")

# Audio file upload
st.header("Upload Audio File")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('Processing...'):
        audio_path = f"/tmp/{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        audio, sr = librosa.load(audio_path, sr=16000)
        transcription = transcribe_audio(audio, sr)
        st.write("Transcribed Text:", transcription)
        
        translation = translate_text(transcription, translation_tokenizer, translation_model)
        st.write("Translated Text:", translation)
        
        synthesized_audio = synthesize_speech(translation)
        
        synthesized_audio_path = f"/tmp/synthesized_{uploaded_file.name}"
        synthesized_audio.export(synthesized_audio_path, format="wav")
        
        st.audio(synthesized_audio_path, format="audio/wav")
