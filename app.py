import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np

# Load the models
model1 = load_model('pets.keras')  # Assuming this model expects (224, 224, 3)
model2 = load_model('classifier.keras')  # Assuming this model expects (150, 150, 3)

# Define the classes for the models
animal_classes = ['Cat', 'Dog']  # Assuming model2 is a binary classifier for Cats and Dogs
emotion_classes = ['Happy', 'Sad', 'Angry']  # Replace with actual classes used in model1

def preprocess_image(image, target_size):
    """ Preprocess the image for the model """
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

def classify_image(image):
    """ Classify the image using both models """
    # Preprocess the image for model1 (assuming it requires 224x224)
    processed_image_model1 = preprocess_image(image, (224, 224))

    # Preprocess the image for model2 (assuming it requires 150x150)
    processed_image_model2 = preprocess_image(image, (150, 150))

    # Predict the animal type
    animal_pred = model2.predict(processed_image_model2)
    animal_class = animal_classes[np.argmax(animal_pred)]

    # Predict the emotion
    emotion_pred = model1.predict(processed_image_model1)
    emotion_class = emotion_classes[np.argmax(emotion_pred)]

    return animal_class, emotion_class

# Streamlit app
st.title("Animal and Emotion Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    animal_class, emotion_class = classify_image(image)

    st.write(f"Animal: {animal_class}")
    st.write(f"Emotion: {emotion_class}")
