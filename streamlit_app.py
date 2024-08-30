%%writefile app3.py
import streamlit as st
import os
import google.generativeai as genai
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
CLASS_LABELS = ["Bacterial Spot","Early Blight","Healthy","Iron Deficiency","Late Blight","Leaf Mold","Leaf Miner","Mosaic Virus","Septoria","Spider Mites","Yellow Leaf Curl Virus"]

# Path to the file in Google Drive
api_key_path = '/content/api_key.txt'

with open(api_key_path) as f:
    api_key = f.read().strip()

# genai.configure(api_key="GOOGLE_API_KEY")
# Function to load YOLO model
def load_yolo_model():
    model = YOLO('/content/best.pt')  # Update the path to your model file
    return model

yolo_model = load_yolo_model()

# Function to load Gemini Pro model and get responses
def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    return response

# Initialize the Streamlit app
st.set_page_config(page_title="AgriChat - YOLOv8 Object Detection & AI Chat")
background_image_url = "https://media.istockphoto.com/id/844226534/photo/leaf-background.jpg?s=612x612&w=0&k=20&c=N4NPPNXFU5hPcThEbQ-wr4y64pqSKm-x5AMDZ0sPL5w="

page_bg_img = f""" <style>
[data-testid="stAppViewContainer"] {{
background-image: url("{background_image_url}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0, 0, 0, 0);
}}

[data-testid="stSidebar"] {{
background: rgba(0, 0, 0, 0.3);
}}
</style>
""

# Inject the CSS into the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown( f""
    <style>
    .centered-header {
        text-align: center;
        font-size: 5em;  /* Adjust size as needed */
        font-weight: bold;
    }
    .sub-header {
        font-size: 50px;  /* Adjust size as needed */
        font-weight: bold; 
    }
    </style> "", unsafe_allow_html=True)

st.markdown('<h1 class="centered-header">AgriChat</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="sub-header">Your one stop destination for all information related to Plant diseases and solutions for its treatment.</h1>', unsafe_allow_html=True)

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# YOLOv8 Object Detection Section
st.subheader("YOLOv8 Object Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to a format suitable for OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform inference
    results = yolo_model(image_cv)

    # Render results on image
    img_with_boxes = results[0].plot()

    # Convert BGR back to RGB for display
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Display the output
    st.image(img_with_boxes, caption="YOLOv8 Object Detection Output", use_column_width=True)
    st.subheader("Detection Results")
    for result in results:
        for box in result.boxes:
            label_index = int(box.cls.item())
            label_name = CLASS_LABELS[label_index]  # Get the label name
            confidence = box.conf.item()  # Convert to a regular Python float
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            st.write(f"Label: {label_name}, Confidence: {confidence:.2f}, "
                     f"BBox Coords: ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})")

# Gemini AI Chat Section
st.subheader("Gemini AI Chat")

input_question = st.text_input("Input your question about plant pesticides or any other query:", key="input")
submit = st.button("Ask the question")

if submit and input_question:
    response = get_gemini_response(input_question)
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", input_question))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))

# Display the chat history
st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
