import streamlit as st 
import pandas as pd 
import numpy as np 
import cv2 
from keras.models import load_model
import io
import os
from PIL import Image, ImageOps
import time 
def main():
    # title block
    st.markdown(
        """
        <div style='text-align: center; background-color: #005f69;'>
            <img src='https://www.ueh.edu.vn/images/logo-header.png' alt='Logo UEH'/>
            <h2>University of Economics Ho Chi Minh City – UEH</h2>
            <h3><img src='https://ctd.ueh.edu.vn/wp-content/uploads/2023/07/cropped-TV_trang_CTD.png' alt='Logo CTD UEH' width='100'/>UEH College Of Technology And Design</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    # end title
    # header block
    st.header("Apply Deep Learning Models for Disease Detection on X-ray Images")
    # end header
    
    # process
    # đưa về dạng thập phân
    np.set_printoptions(suppress=True)
    
    # xử lý chọn model
    menu = ["Home", "Model 1", "Model 2", "Model 3"]
    st.sidebar.title('Navigation')
    choice = st.sidebar.selectbox("Choose a Model", menu)
    isLoaded = False
    pixels = 0
    class_name_labels = None
    if choice == "Home":
        st.write("Choose a model in the left Navigation first!!!!")
    elif choice == "Model 1":
        st.subheader("Model 1")
        st.write("Model 1: Training using dataset Chest X-Ray Images (Pneumonia) [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)")
        st.write("The model will be able to detect two classifications: normal and pneumonia.")
        # load model
        with st.spinner("Loading Model..."):
            model = load_model("tm_models/dataset1/keras_model.h5", compile=False)
            isLoaded = True
            pixels = 224
            class_name_labels = open("tm_models/dataset1/labels.txt", "r").readlines()
    elif choice == "Model 2":
        st.subheader("Model 2")
        st.write("Model 2: Training using dataset Chest X-ray (Covid-19 & Pneumonia) [here](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)")
        st.write("The model will be able to detect three classifications: covid, normal, and pneumonia.")
        # load model
        with st.spinner("Loading Model..."):
            model = load_model("tm_models/dataset2/keras_model.h5", compile=False)
            isLoaded = True
            pixels = 224
            class_name_labels = open("tm_models/dataset2/labels.txt", "r").readlines()
    elif choice == "Model 3":
        st.subheader("Model 3")
        st.write("Model 3: Training using dataset Curated Chest X-Ray Image Dataset for COVID-19  [here](https://www.kaggle.com/datasets/unaissait/curated-chest-xray-image-dataset-for-covid19)")
        st.write("The model will be able to detect 4 classifications: covid, normal, bacterial pneumonia and viral pneumonia.")
        # load model
        with st.spinner("Loading Model..."):
            model = load_model("tm_models/dataset3/keras_model.h5", compile=False)
            isLoaded = True
            pixels = 224
            class_name_labels = open("tm_models/dataset3/labels.txt", "r").readlines()
    else: st.write("Do not choose model yet!!!")

    # upload file
    if isLoaded:
        uploaded_image = st.file_uploader("Choose an x-ray image")
        data = np.ndarray(shape=(1, pixels, pixels, 3), dtype=np.float32)
        class_names = class_name_labels
        if not (uploaded_image is None):
            # get image info and show
            img_cap = "File size: " + str(uploaded_image.size) + " kb"
            st.image(uploaded_image, caption=img_cap)
            
            image_data = uploaded_image.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            # resize image 
            size = (pixels, pixels)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            # turn the image into a numpy array
            image_array = np.asarray(image)
            # normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            # image into the array
            data[0] = normalized_image_array
            # predict image
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            st.info("Classification report: ")
            st.write("Class:", class_name[2:])
            st.write("Confidence Score: ", confidence_score)
            st.info("Prediction result: ")
            
            if class_name[2:] == "COVID19\n":
                percentx = round(confidence_score,2) * 100
                st.error("The uploaded X-ray film is predicted to have a risk of Covid-19 with an accuracy of " + str(percentx) + "%. You should perform specialized tests at the nearest medical facility as soon as possible.")
            elif class_name[2:] == "PNEUMONIA\n":
                percentx = round(confidence_score,2) * 100
                st.error("The uploaded X-ray film is predicted to have a risk of pneumonia with an accuracy of " + str(percentx) + "%. You should perform specialized tests at the nearest medical facility as soon as possible.")
            elif class_name[2:] == "PNEUMONIABACTERIAL\n":
                percentx = round(confidence_score,2) * 100
                st.error("The uploaded X-ray film is predicted to have a risk of bacterial pneumonia with an accuracy of " + str(percentx) + "%. You should perform specialized tests at the nearest medical facility as soon as possible.")
            elif class_name[2:] == "PNEUMONIAVIRAL\n":
                percentx = round(confidence_score,2) * 100
                st.error("The uploaded X-ray film is predicted to have a risk of viral pneumonia with an accuracy of " + str(percentx) + "%. You should perform specialized tests at the nearest medical facility as soon as possible.")
            elif class_name[2:] == "NORMAL\n":
                percentx = round(confidence_score,2) * 100
                st.success("The model does not predict any abnormalities, but if you have ANY SYMPTOMS of pneumonia, please go to the nearest medical facility immediately.")
            else:
                st.success("The model does not predict any abnormalities, but if you have ANY SYMPTOMS of pneumonia, please go to the nearest medical facility immediately.")
    
def samples_h():
    st.subheader("Get some image samples: ")
    st.write("You can also download the COVID-19 Detection X-Ray Dataset [here](https://www.kaggle.com/datasets/darshan1504/covid19-detection-xray-dataset), which we do not use for training")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image_sample_xray/normal_1.jpeg", caption="Normal Xray Image 1")
        with open("image_sample_xray/normal_1.jpeg", "rb") as file:
            st.download_button(
                label = "Download Normal Xray Image 1",
                data = file,
                file_name = "normal_1.jpeg",
                mime = "image/jpeg"
            )
    with col2:
        st.image("image_sample_xray/pneumonia_viral_1.jpeg", caption="Viral Pneumonia Xray Image 1")
        with open("image_sample_xray/pneumonia_viral_1.jpeg", "rb") as file:
            st.download_button(
                label = "Download Viral Pneumonia Xray Image 1",
                data = file,
                file_name = "pneumonia_viral_1.jpeg",
                mime = "image/jpeg"
            )
    with col3:
        st.image("image_sample_xray/pneumonia_viral_2.jpeg", caption="Viral Pneumonia Xray Image 2")
        with open("image_sample_xray/pneumonia_viral_2.jpeg", "rb") as file:
            st.download_button(
                label = "Download Viral Pneumonia Xray Image 2",
                data = file,
                file_name = "pneumonia_viral_2.jpeg",
                mime = "image/jpeg"
            )    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.image("image_sample_xray/covid_1.jpeg", caption="Normal Xray Image 1")
        with open("image_sample_xray/covid_1.jpeg", "rb") as file:
            st.download_button(
                label = "Download Covid Xray Image 1",
                data = file,
                file_name = "covid_1.jpeg",
                mime = "image/jpeg"
            )
    with col5:
        st.image("image_sample_xray/pneumonia_bacterial_1.jpeg", caption="Bacterial Pneumonia Xray Image 1")
        with open("image_sample_xray/pneumonia_bacterial_1.jpeg", "rb") as file:
            st.download_button(
                label = "Download Bacterial Pneumonia Xray Image 1",
                data = file,
                file_name = "pneumonia_bacterial_1.jpeg",
                mime = "image/jpeg"
            )
    with col6:
        st.image("image_sample_xray/pneumonia_bacterial_2.jpeg", caption="Bacterial Pneumonia Xray Image 2")
        with open("image_sample_xray/pneumonia_bacterial_2.jpeg", "rb") as file:
            st.download_button(
                label = "Download Bacterial Pneumonia Xray Image 2",
                data = file,
                file_name = "pneumonia_bacterial_2.jpeg",
                mime = "image/jpeg"
            )
    
def footer_h():
    st.subheader("Information of this Website")
    st.warning("These models are trained for educational purposes with limited data (small amount of samples) and have not been verified by experts for accuracy!!")
    st.write("All models using the Teachable Machine web-based tool [here](https://teachablemachine.withgoogle.com)")
    

        
if __name__ == "__main__":
    main()
    samples_h()
    footer_h()
