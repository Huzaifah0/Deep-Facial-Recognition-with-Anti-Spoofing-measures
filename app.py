# app.py

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import pickle
import onnxruntime as ort
from io import BytesIO
import model

# Load Face Recognition and Anti-Spoofing models
@st.cache_resource
def load_models():
    face_recog_model=model.InceptionResnetV1()
    face_recog_model.load_state_dict(torch.load('models/face_recog.pt', map_location='cpu'))
    print(face_recog_model)
    face_recog_model.eval()
    spoof_sess = ort.InferenceSession('models/anti-spoof-mn3.onnx')
    return face_recog_model, spoof_sess

face_model, spoof_model = load_models()

# Load or initialize user embeddings database
def load_user_embeddings():
    try:
        with open('embeddings/users_embeddings.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_user_embeddings(user_embeddings):
    with open('embeddings/users_embeddings.pkl', 'wb') as f:
        pickle.dump(user_embeddings, f)

# Face detection using OpenCV DNN
def detect_faces_opencv_dnn(image):
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append(box.astype(int))
    return boxes

# Get face embedding
def get_face_embedding(image):
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding = face_model(image_tensor).squeeze(0)
    return embedding

# Anti-spoofing prediction
def is_real_face(face_image):
    resized = cv2.resize(face_image, (224, 224)).astype(np.float32)
    input_blob = resized.transpose(2, 0, 1)[np.newaxis, :]
    input_blob /= 255.0
    ort_inputs = {spoof_model.get_inputs()[0].name: input_blob}
    output = spoof_model.run(None, ort_inputs)[0]
    prediction = np.argmax(output)
    return prediction == 1  # 1 = real, 0 = spoof

# Add user to the embeddings database
def add_user_to_db(image, name, user_embeddings):
    embedding = get_face_embedding(image)
    if embedding is not None:
        user_embeddings[name] = embedding.numpy()
        save_user_embeddings(user_embeddings)

# Recognize faces
def recognize_faces(image, user_embeddings):
    boxes = detect_faces_opencv_dnn(image)
    recognized = []
    for box in boxes:
        x1, y1, x2, y2 = box
        face = image[y1:y2, x1:x2]
        if is_real_face(face):
            embedding = get_face_embedding(face)
            min_dist = float('inf')
            name = "Unknown"
            for user_name, user_embedding in user_embeddings.items():
                dist = np.linalg.norm(embedding.numpy() - user_embedding)
                if dist < min_dist and dist < 0.9:  # threshold
                    min_dist = dist
                    name = user_name
            recognized.append((box, name))
    return recognized

# Streamlit interface
st.title("Face Recognition with Anti-Spoofing")
st.sidebar.title("Options")

user_embeddings = load_user_embeddings()

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Uploaded Image", use_column_width=True)
    name = st.sidebar.text_input("Enter name for this image")

    if st.sidebar.button("Add to Database"):
        if name:
            add_user_to_db(image_np, name, user_embeddings)
            st.sidebar.success(f"User {name} added!")
        else:
            st.sidebar.warning("Please enter a name.")

st.sidebar.title("Webcam")
run_webcam = st.sidebar.button("Start Webcam")

if run_webcam:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognize_faces(frame_rgb, user_embeddings)

        for (x1, y1, x2, y2), name in results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.image(frame, channels="BGR", use_column_width=True)

    cap.release()
