import streamlit as st
import whisper
import librosa
import os
import ffmpeg as ffmpeg

st.title("Fr_Ln Transcription and traduction using whisper")

# load model
small_model = whisper.load_model("small")
st.text("Whisper model loaded")

# upload audio file with streamlit
audio_file = st.file_uploader("upload audio", type=["wav", "mp3", "m4a"])
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Convert the uploaded file to WAV format
    try:
        audio_bytes = audio_file.read()
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)

        converted_audio, sr = librosa.load("output.wav", sr=16000)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        # return

    my_audio = whisper.load_audio(converted_audio)
    result = small_model.transcribe(my_audio)
    st.write(result["text"])

    # Clean up the temporary file
    os.remove("output.wav")
