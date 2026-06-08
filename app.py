import streamlit as st
from roboflow import Roboflow
from PIL import Image
import os

st.title("Bus EDS Monitor")

# Initialize Roboflow (Use your NEW, secure API key)
rf = Roboflow(api_key="x0iljMh1cc1LteBJrWfr") 
project = rf.workspace("syazwis-workspace").project("eds-2")
model = project.version(4).model

# Add Camera input option
source = st.radio("Select Source:", ("Upload Image", "Use Camera"))

if source == "Use Camera":
    uploaded_file = st.camera_input("Take a photo of the EDS")
else:
    uploaded_file = st.file_uploader("Upload Bus Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file locally temporarily
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
  # Run Inference
    prediction = model.predict(temp_path)
    
    # Check if we have detections to avoid errors
    if prediction:
        # Save the annotated image directly
        # Some versions of roboflow save the plot as an image object
        annotated_image = prediction.plot()
        
        # Save directly using the built-in method
        # If .save() is not available on the returned object, 
        # we can save the prediction visualization directly
        prediction.save("result.jpg") 
        
        # Display results
        st.image("result.jpg", caption='Result', use_container_width=True)
        
        # Add Download Button for the result
        with open("result.jpg", "rb") as file:
            st.download_button(label="Save Result", data=file, file_name="eds_result.jpg", mime="image/jpeg")
    else:
        st.write("No EDS detected in this image.")
