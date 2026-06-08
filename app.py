import streamlit as st
from roboflow import Roboflow
from PIL import Image
import os

st.title("Bus EDS Monitor")

# Initialize Roboflow
# IMPORTANT: Use your new API key here!
rf = Roboflow(api_key="x0iljMh1cc1LteBJrWfr") 
project = rf.workspace("syazwis-workspace").project("eds-2")

# Update this to your latest trained version (e.g., version 5)
model = project.version(5).model 

# Camera vs Upload option
source = st.radio("Select Source:", ("Upload Image", "Use Camera"))

if source == "Use Camera":
    uploaded_file = st.camera_input("Take a photo of the EDS")
else:
    uploaded_file = st.file_uploader("Upload Bus Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Run Inference with confidence and overlap filters
    prediction = model.predict(temp_path, confidence=50, overlap=25)
    
    # Check if detection was successful
    if len(prediction.json()['predictions']) > 0:
        prediction.save("result.jpg")
        st.image("result.jpg", caption='Result', use_container_width=True)
        
        # Save Result button
        with open("result.jpg", "rb") as file:
            st.download_button(label="Save Result", data=file, file_name="eds_result.jpg", mime="image/jpeg")
    else:
        st.write("No EDS detected. Ensure the display is clearly in frame.")
