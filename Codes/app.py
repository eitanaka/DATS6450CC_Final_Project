import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import boto3
import os

IMAGE_SIZE = 150 

@st.cache_resource
def load_model():

    os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
    aws_access_key_id = st.secrets["default"]["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["default"]["AWS_SECRET_ACCESS_KEY"]

    # Try
    s3 = boto3.resource(
        service_name='s3',
        region_name=os.environ['AWS_DEFAULT_REGION'],
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    bucket_name = "cc-final-project"
    object_key = "intel_classifier.h5"
    local_model_file = "intel_classifier.h5"

    if not os.path.exists(local_model_file):
        s3.download_file(bucket_name, object_key, local_model_file)

    # Load the model
    model = tf.keras.models.load_model(local_model_file)

    return model

# Cache the loaded model using st.cache_data
model = load_model()

# Set up the Streamlit UI
st.title("Intel Image Classifier with TensorFlow")
st.write("Upload an image and the model will predict its class.")

# Instructions about the classes
st.markdown("""
This model classifies images into the following categories:
- **Buildings**
- **Forest**
- **Glacier**
- **Mountain**
- **Sea**
- **Street**

Please upload an image belonging to one of these categories.
""")

# Upload image
upload_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Process the image
    def preprocess(image):
        # Resize the image
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    processed_image = preprocess(image)

    # Show spinner while making predictions
    with st.spinner('Predicting...'):
        # Make predictions
        prediction = model.predict(processed_image)
    
    # Class names
    class_names = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
    predicted_class = class_names[np.argmax(prediction)]

    # Show main prediction with larger font
    st.markdown(f'## Prediction: {predicted_class}')

    # Display all prediction probabilities as a table with 3 digit precision
    st.write("Prediction Probabilities:")
    prob_df = pd.DataFrame(prediction, columns=class_names).round(3)
    st.table(prob_df)