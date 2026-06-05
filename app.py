import streamlit as st
from roboflow import Roboflow
from PIL import Image

st.title("Bus EDS Monitor")

# Initialize Roboflow with your API key
rf = Roboflow(api_key="x0iljMh1cc1LteBJrWfr") 
project = rf.workspace("syazwis-workspace").project("eds-2")
model = project.version(4).model

# File uploader for mobile/desktop users
uploaded_file = st.file_uploader("Upload Bus Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Bus Display', use_container_width=True)
    
    # Run Inference
    # The model expects an image path, we save the uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    prediction = model.predict("temp.jpg")
    annotated_image = prediction.plot()
    
    # Show the result with bounding boxes
    st.image(annotated_image, caption='Result', use_container_width=True)
    st.write("Detection results displayed above.")
