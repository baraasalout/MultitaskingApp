
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import gc
from gtts import gTTS
import base64
import os

# Function to handle image upload/capture
def handle_image_input():
    st.sidebar.title("Choose Image Input")
    input_type = st.sidebar.radio("How would you like to provide the image?", ("Upload Image", "Capture Image"))
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            return Image.open(uploaded_file)
    elif input_type == "Capture Image":
        captured_image = st.camera_input("Capture an image using your webcam")
        if captured_image is not None:
            return Image.open(captured_image)
    return None

# Function to handle language selection
# Function to handle language selection with unique key
def select_language(key):
    languages = {
        "English": "en",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Italian": "it",
        "Arabic": "ar"
    }
    selected_language = st.sidebar.selectbox("Choose the language for translation", list(languages.keys()), key=key)
    return languages[selected_language]

# Function for text translation
@st.cache_resource
def load_translator(target_language):
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language}", trust_remote_code=True)

# Function to convert text to speech
def text_to_speech(text, language):
    tts = gTTS(text, lang=language)
    audio_file_path = "caption.mp3"
    tts.save(audio_file_path)
    with open(audio_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
    os.remove(audio_file_path)
    return f'<a href="data:audio/mp3;base64,{audio_base64}" download="caption.mp3">Download Caption as MP3</a>'

# Function to detect objects in an image
@torch.no_grad()
def detect_objects(image, threshold=0.5):
    object_detector = pipeline("object-detection", model="hustvl/yolos-tiny", trust_remote_code=True)

    objects = object_detector(image)
    return [obj for obj in objects if obj['score'] >= threshold]


# Function to translate the caption and object detection labels
def translate_text(text, target_language_code):
    if target_language_code == 'en':
        return text  # No translation needed for English
    elif target_language_code == 'ar':
        # Use a specific translation model for English to Arabic
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar", trust_remote_code=True)
        translated_text = translator(text)[0]['translation_text']
    else:
        # Use a general translation model for other languages
        translator = load_translator(target_language_code)
        translated_text = translator(text)[0]['translation_text']

    return translated_text
# Function to generate image caption
@torch.no_grad()
def generate_caption(image):
    caption_generator = pipeline("image-to-text", model="ydshieh/vit-gpt2-coco-en", trust_remote_code=True)
    return caption_generator(image)[0]['generated_text']

# Function for Visual Question Answering
@torch.no_grad()
def visual_question_answering(image, question):
    vqa_model = pipeline("visual-question-answering")
    result = vqa_model(image=image, question=question, top_k=1)
    return result[0]['answer']

# Function for Fill Mask
def fill_mask_task(user_input):
    fill_mask_model =pipeline("fill-mask", model="bert-base-uncased", trust_remote_code=True)
    result = fill_mask_model(user_input)
    return result

# Task-specific functions
def perform_object_detection():
    uploaded_image = handle_image_input()
    if uploaded_image:
        resized_image = uploaded_image.resize((512, 512))
        st.image(resized_image, caption="Uploaded/Captured Image (Resized)", use_column_width=True)
        with st.spinner("Detecting objects..."):
            objects = detect_objects(resized_image)
            for obj in objects:
                st.write(f"{obj['label']} with confidence {obj['score']:.2f}")


# Example usage in tasks:
def perform_image_captioning():
    uploaded_image = handle_image_input()
    if uploaded_image:
        resized_image = uploaded_image.resize((512, 512))
        st.image(resized_image, caption="Uploaded/Captured Image (Resized)", use_column_width=True)
        with st.spinner("Generating caption..."):
            caption = generate_caption(resized_image)
            st.write(f"Caption: {caption}")
            translated_caption = translate_text(caption, select_language(key="caption_lang"))
            st.write(f"Translated Caption: {translated_caption}")
            # st.markdown(text_to_speech(translated_caption, select_language(key="caption_tts_lang")), unsafe_allow_html=True)

def perform_sentiment_analysis():
    user_input = st.text_input("Enter a sentence for sentiment analysis:")
    if user_input:
        sentiment_analyzer =pipeline("sentiment-analysis", trust_remote_code=True)
        result = sentiment_analyzer(user_input)
        st.write(f"Sentiment: {result[0]['label']} (Confidence: {result[0]['score']:.2f})")


#
# @st.cache_resource
# def load_fill_mask():
#     return pipeline("fill-mask", model="bert-base-uncased", trust_remote_code=True)
#

def perform_fill_mask():
    user_input = st.text_input("Enter a sentence with a [MASK] token:")
    if user_input:
        with st.spinner("Filling the mask..."):
            result = fill_mask_task(user_input)
            st.write("Possible completions:")
            for r in result:
                st.write(f"- {r['sequence']} (Confidence: {r['score']:.2f})")

def perform_visual_question_answering():
    uploaded_image = handle_image_input()
    if uploaded_image:
        resized_image = uploaded_image.resize((512, 512))
        st.image(resized_image, caption="Uploaded/Captured Image (Resized)", use_column_width=True)
        question = st.text_input("Enter your question about the image:")
        if question:
            with st.spinner("Answering your question..."):
                answer = visual_question_answering(uploaded_image, question)
                st.write(f"Answer: {answer}")

def perform_table_question_answering():
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        table = pd.read_csv(uploaded_file)
        table = table.astype(str)
        st.write("Table:")
        st.dataframe(table)

        query = st.text_input("Enter your question about the table:")
        if query:
            with st.spinner("Answering your question..."):
                tqa_model = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")
                result = tqa_model(table=table, query=query)
                st.write(f"Predicted answer: {result['cells'][0]}")

# Main app logic
st.title("Multitasking App")
task = st.sidebar.selectbox("Select a task", ("Object Detection", "Image Captioning" ,"Visual Question Answering", "Sentiment Analysis", "Fill Mask", "Table Question Answering"))

if task == "Object Detection":
    perform_object_detection()
elif task == "Image Captioning":
    perform_image_captioning()
elif task == "Sentiment Analysis":
    perform_sentiment_analysis()
elif task == "Fill Mask":
    perform_fill_mask()
elif task == "Visual Question Answering":
    perform_visual_question_answering()
elif task == "Table Question Answering":
    perform_table_question_answering()

# Call garbage collection at the end to free up memory
gc.collect()
