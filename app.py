import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn_model.h5')

def preprocess_image(image):
    """Preprocess uploaded image for MNIST model"""
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Invert colors (MNIST has white digits on black background)
    image = 255 - image
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Reshape for model
    image = image.reshape(1, 28, 28, 1)
    
    return image

# Streamlit App
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) and get a prediction!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Preprocess the image
    processed_image = preprocess_image(image_array)
    
    # Load model and make prediction
    model = load_model()
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Display results
    st.write(f"**Predicted Digit: {predicted_digit}**")
    st.write(f"**Confidence: {confidence:.2f}%**")
    
    # Show prediction probabilities
    st.write("Prediction Probabilities:")
    prob_df = pd.DataFrame({
        'Digit': range(10),
        'Probability': prediction[0]
    })
    st.bar_chart(prob_df.set_index('Digit'))

# Drawing canvas (optional enhancement)
st.write("---")
st.write("Or draw a digit below:")

# You can add a drawing canvas using streamlit-drawable-canvas
# pip install streamlit-drawable-canvas
