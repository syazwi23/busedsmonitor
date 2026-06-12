import streamlit as st
from roboflow import Roboflow
from PIL import Image
import os

st.markdown("""
    <div style="background-color:#004a99; padding:20px; border-radius:10px; text-align:center; margin-bottom: 20px; border: 2px solid white;">
        <h1 style="color:white; font-family:sans-serif; font-weight:bold; margin:0; letter-spacing: 1px;">SBS TRANSIT</h1>
        <p style="color:white; font-size:16px; margin:0; font-style:italic;">EDS INSPECTION SYSTEM</p>
    </div>
""", unsafe_allow_html=True)

st.title("Bus EDS Monitor")

# Initialize Roboflow
rf = Roboflow(api_key="x0iljMh1cc1LteBJrWfr") 
project = rf.workspace("syazwis-workspace").project("eds-2")
model = project.version(5).model # Ensure this is your latest version

# Camera input only
uploaded_file = st.camera_input("Take a photo of the EDS")

if uploaded_file is not None:
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Run Inference with optimized settings
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
