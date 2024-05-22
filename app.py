import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import pickle


# model_path = 'C:\\Users\\Dell\\Machine learning\\signature.pkl'
# with open(model_path, 'rb') as model_file:
#     model = pickle.load(model_file)

model = tf.keras.models.load_model(r'C:\Users\Dell\Machine learning\signature.h5')

st.title("Signature recogination")

# File upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.convert('L')
    img = img.resize((256,256))
    img = np.expand_dims(img, axis=0) / 255.0

    output = model.predict(img)
    output = output.argmax(axis = 1)


    if int(output[0]) == 0:
        original_title = '<p style="font-size: 20px;">Signature is fake</p>'
        st.markdown(original_title, unsafe_allow_html=True)
    else:
        original_title = '<p style="font-size: 20px;">Signature is original</p>'
        st.markdown(original_title, unsafe_allow_html=True)
