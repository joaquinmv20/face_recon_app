import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64
import face_recognition as fr
import io
import cmake
import dlib

st.set_page_config(
    layout="wide", 
    page_title="Face Recognition App", 
    page_icon=":camera:")


st.markdown("<h1 style='font-size: 80px; margin-bottom: 0;'>Recog<span style='color: orange;'>NICE!</span> 📷 👨‍🦰</h1>", unsafe_allow_html=True)
st.markdown("<hr style='height:2px;border-width:0;color:black;background-color:black;margin-top: 0;'>", unsafe_allow_html=True)

st.write("## Upload :gear:")
MAX_FILE_SIZE = 5 * 1024 * 1024 


col1, col2 = st.columns(2)
my_upload = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        image = Image.open(io.BytesIO(my_upload.read()))  # Convert the image to the format the app needs to work
        image = image.resize((300, 300))  # Resize the image
        st.image(image)  # Display the image

        df = fr.search_in_database(image)  # Search for a face in the existing database

        if df.empty or df['Result'].iloc[0] == "Different persons":
            st.markdown("<h1 style='font-size: 30px;'>Your image is not in the system. Please contact support</h1>", unsafe_allow_html=True)
        else:
            input_string = df['Image'].iloc[0]
            partes = input_string.split('.')
            partes_importantes = partes[0].split('_')
            resultado = partes_importantes[0] + ' ' + partes_importantes[1]
            st.markdown(f"<h1 style='font-size: 80px;'>{resultado}</h1>", unsafe_allow_html=True)  # Display the result with larger font size
