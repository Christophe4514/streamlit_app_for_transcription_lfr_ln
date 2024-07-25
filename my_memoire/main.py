import streamlit as st
import whisper
import os
# import sounddevice as sd
from deep_translator import GoogleTranslator

st.title("Fr_Ln Transcription and traduction using whisper")

# prompt a text to translate
text_to_translate = st.text_input("Write the text to translate")

# upload audio to transcribe and translate late
audio_file = st.file_uploader("upload audio", type=["wav", "mp3", "m4a"])


# print(audio_file.name)
# load model
small_model = whisper.load_model("small")
st.text("Whisper model loaded")

# load audio
if audio_file is not None:
    try:
        audio = audio_file.name
        transcribe = whisper.load_audio(audio)
    except FileNotFoundError:
        print("Erreur : Fichier audio introuvable.")
    except RuntimeError as e:
        print("Erreur lors du chargement de l'audio :", e)
else:
    st.sidebar.error("Please upload an audio file")

# transcribe audio using whisper model
if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")
        # audio_file = whisper.load_audio(audio_file.name)
        transcription = small_model.transcribe(audio, fp16=False)
        st.sidebar.success("Transcription completed")
        st.markdown(transcription["text"])
    else:
        st.sidebar.error("Please upload an audio file")

# translate audio using google on whisper model
if st.sidebar.button("Translate"):
    translator = GoogleTranslator(source='auto', target='ln')
    prompt = ('Au commencement, Dieu créa les cieux et la terre. La terre était informe et vide: il y avait des '
              'ténèbres à la surface de l\'abîme, et les ténèbres étaient à la surface de l\'abîme, et l\'Esprit de '
              'Dieu se mouvait sur les eaux.')
    prompt_2 = "Hello, it's me, I was wondering if I was going to be able to do it. I'm going to be able to do it"
    transcribed_text = prompt_2
    translated = translator.translate(transcribed_text)
    st.sidebar.success("Tanslation completed")
    st.markdown(translated)

# translate text prompted using google on whisper model
if st.sidebar.button("Translate the text prompted"):
    translator = GoogleTranslator(source='auto', target='ln')
    transcribed_text = text_to_translate
    translated = translator.translate(transcribed_text)
    st.sidebar.success("Tanslation completed")
    st.markdown(translated)

st.sidebar.header("Play Original Audio")
st.sidebar.audio(audio_file)
