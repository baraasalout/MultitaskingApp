import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import gc
from gtts import gTTS  # Text-to-Speech library
import base64
import os
from transformers import TapasTokenizer, TapasForQuestionAnswering
# Load the TAPAS tokenizer and model
@st.cache_resource
def load_tapas_pipeline():
    return pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

# Load Hugging Face models with trust_remote_code to avoid warnings
@st.cache_resource
def load_object_detector():
    return pipeline("object-detection", model="hustvl/yolos-tiny", trust_remote_code=True)


@st.cache_resource
def load_caption_generator():
    return pipeline("image-to-text", model="ydshieh/vit-gpt2-coco-en", trust_remote_code=True)


@st.cache_resource
def load_sentiment_analysis():
    return pipeline("sentiment-analysis", trust_remote_code=True)


@st.cache_resource
def load_zero_shot_classifier():
    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",  # Specify the model here
        trust_remote_code=True
    )


@st.cache_resource
def load_fill_mask():
    return pipeline("fill-mask", model="bert-base-uncased", trust_remote_code=True)


@st.cache_resource
def load_translator(target_language):
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language}", trust_remote_code=True)


# Disable gradients to save memory
@torch.no_grad()
def detect_objects(image, object_detector, threshold=0.5):
    image = image.convert("RGB")
    objects = object_detector(image)
    filtered_objects = [obj for obj in objects if obj['score'] >= threshold]  # Filter by threshold
    return filtered_objects


@torch.no_grad()
def generate_caption(image, caption_generator):
    image = image.convert("RGB")
    caption = caption_generator(image)[0]['generated_text']
    return caption


# Function to convert text to speech and return a download link
def text_to_speech(text, language):
    tts = gTTS(text, lang=language)
    audio_file_path = "caption.mp3"
    tts.save(audio_file_path)

    # Load and encode the audio file
    with open(audio_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_link = f'<a href="data:audio/mp3;base64,{audio_base64}" download="caption.mp3">Download Caption as MP3</a>'

    # Remove the audio file after reading
    os.remove(audio_file_path)
    return audio_link


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


# Streamlit app
st.title("Multitasking App")

# Sidebar for input
st.sidebar.title("Choose Task")
task = st.sidebar.selectbox(
    "Select a task",
    ("Object Detection", "Image Captioning", "Sentiment Analysis", "Zero-Shot Classification", "Fill Mask", "Table Question Answering","Visual Question Answering")
)

# Object Detection Task
if task == "Object Detection":
  
    if task in ["Object Detection", "Image Captioning"]:
        st.sidebar.title("Choose Image Input")
        input_type = st.sidebar.radio("How would you like to provide the image?", ("Upload Image", "Capture Image"))
    
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    uploaded_image = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    
        if input_type == "Capture Image":
            captured_image = st.camera_input("Capture an image using your webcam")
            if captured_image is not None:
                try:
                    uploaded_image = Image.open(captured_image)
                except Exception as e:
                    st.error(f"Error capturing image: {e}")


    # Language selection
    st.sidebar.title("Select Language")
    languages = {
        "English": "en",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Italian": "it",
        "Arabic": "ar"
    }
    selected_language = st.sidebar.selectbox("Choose the language for the caption", list(languages.keys()))
    selected_language_code = languages[selected_language]

    resized_image = uploaded_image.resize((512, 512))  # Resize early to save memory
    st.image(resized_image, caption="Uploaded/Captured Image (Resized)", use_column_width=True)

    confidence_threshold = st.slider("Set object detection confidence threshold", 0.1, 1.0, 0.5)

    with st.spinner("Detecting objects..."):
        try:
            object_detector = load_object_detector()
            objects = detect_objects(resized_image, object_detector, threshold=confidence_threshold)
            st.write("Objects detected:")
            translated_labels = []
            for obj in objects:
                translated_label = translate_text(obj['label'], selected_language_code)
                translated_labels.append(translated_label)
                st.write(f"- {translated_label} with confidence {obj['score']:.2f}")
        except Exception as e:
            st.error(f"Error in object detection: {e}")

# Image Captioning Task
elif task == "Image Captioning":
        
    if task in ["Object Detection", "Image Captioning"]:
        st.sidebar.title("Choose Image Input")
        input_type = st.sidebar.radio("How would you like to provide the image?", ("Upload Image", "Capture Image"))
    
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    uploaded_image = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    
        if input_type == "Capture Image":
            captured_image = st.camera_input("Capture an image using your webcam")
            if captured_image is not None:
                try:
                    uploaded_image = Image.open(captured_image)
                except Exception as e:
                    st.error(f"Error capturing image: {e}")
    resized_image = uploaded_image.resize((512, 512))
    st.image(resized_image, caption="Uploaded/Captured Image (Resized)", use_column_width=True)
        # Language selection
    st.sidebar.title("Select Language")
    languages = {
        "English": "en",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Italian": "it",
        "Arabic": "ar"
    }
    selected_language = st.sidebar.selectbox("Choose the language for the caption", list(languages.keys()))
    selected_language_code = languages[selected_language]

    resized_image = uploaded_image.resize((512, 512))  # Resize early to save memory
    st.image(resized_image, caption="Uploaded/Captured Image (Resized)", use_column_width=True)
    with st.spinner("Generating and translating caption..."):
        try:
            caption_generator = load_caption_generator()
            caption = generate_caption(resized_image, caption_generator)
            translated_caption = translate_text(caption, selected_language_code)
            st.write(f"Image Caption in {selected_language}: {translated_caption}")
            st.markdown(text_to_speech(translated_caption, selected_language_code), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error in caption generation or translation: {e}")

# Sentiment Analysis Task
elif task == "Sentiment Analysis":
    user_input = st.text_input("Enter a sentence for sentiment analysis:")
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            try:
                sentiment_analyzer = load_sentiment_analysis()
                result = sentiment_analyzer(user_input)
                st.write(f"Sentiment: {result[0]['label']} (Confidence: {result[0]['score']:.2f})")
            except Exception as e:
                st.error(f"Error in sentiment analysis: {e}")

# Zero-Shot Classification Task
elif task == "Zero-Shot Classification":
    user_input = st.text_input("Enter a sentence for classification:")
    candidate_labels = st.text_input("Enter candidate labels separated by commas:")
    if user_input and candidate_labels:
        candidate_labels = [label.strip() for label in candidate_labels.split(",")]
        with st.spinner("Classifying..."):
            try:
                classifier = load_zero_shot_classifier()
                result = classifier(user_input, candidate_labels)
                st.write("Classification results:")
                for label, score in zip(result['labels'], result['scores']):
                    st.write(f"- {label}: {score:.2f}")
            except Exception as e:
                st.error(f"Error in zero-shot classification: {e}")

# Fill Mask Task
elif task == "Fill Mask":
    user_input = st.text_input("Enter a sentence with a [MASK] token:")
    if user_input:
        with st.spinner("Filling the mask..."):
            try:
                fill_mask_model = load_fill_mask()
                result = fill_mask_model(user_input)
                st.write("Possible completions:")
                for r in result:
                    st.write(f"- {r['sequence']} (Confidence: {r['score']:.2f})")
            except Exception as e:
                st.error(f"Error in fill-mask task: {e}")
# Task for Table Question Answering
elif task == "Table Question Answering":
    st.sidebar.title("Table Question Answering")

    # Upload CSV for table input
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        table = pd.read_csv(uploaded_file)
        table = table.astype(str)

        st.write("Table:")
        st.dataframe(table)

        # User input for question
        query = st.text_input("Enter your question about the table:")

        if query:
            with st.spinner("Answering your question..."):
                try:
                    # Load the TAPAS pipeline
                    tqa = load_tapas_pipeline()

                    # Get the result from TAPAS
                    result = tqa(table=table, query=query)

                    # Check if any answers are returned
                    if 'cells' in result and len(result['cells']) > 0:
                        # Display the predicted answer
                        st.write(f"Predicted answer: {result['cells'][0]}")
                    else:
                        st.write("No answer found for the question.")

                except Exception as e:
                    st.error(f"Error in answering the table question: {e}")

elif task == "Visual Question Answering":
    if task in ["Object Detection", "Image Captioning"]:
        st.sidebar.title("Choose Image Input")
        input_type = st.sidebar.radio("How would you like to provide the image?", ("Upload Image", "Capture Image"))
    
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    uploaded_image = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    
    # File uploader for image input
    uploaded_image = st.file_uploader("Upload your document image (PNG, JPG, etc.)", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Document", use_column_width=True)

        # Input question from the user
        question = st.text_input("Enter your question about the document:")

        if question:
            try:
                # Load the LayoutLM model using pipeline for document question answering
                doc_pipe = pipeline("visual-question-answering")

                # Get the result from the model
                result = doc_pipe(image=image, question=question, top_k=1)
                # vqa_pipeline(image, question, top_k=1)

                # Display the answer
                st.write("Answer:")
                st.write(result[0]['answer'])
            except Exception as e:
                st.error(f"Error in answering the document question: {e}")
else:
    st.warning("Please select a task and provide the required input.")



# Call garbage collection at the end to free up memory
gc.collect()
