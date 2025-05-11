import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import pickle
import onnxruntime as ort
from facenet_pytorch import InceptionResnetV1

@st.cache_resource
def load_models():
    face_recog_model = InceptionResnetV1(pretrained='vggface2').eval()
    spoof_session = ort.InferenceSession('models/anti-spoof-mn3.onnx')
    return face_recog_model, spoof_session

face_model, spoof_model = load_models()

@st.cache_resource
def load_face_detector():
    return cv2.FaceDetectorYN.create(
        model='models/face_detection_yunet_2023mar.onnx',
        config='',
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=500
    )

face_detector = load_face_detector()

def load_user_embeddings():
    try:
        with open('embeddings/users_embeddings.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Create the file if not found and return an empty dictionary
        with open('embeddings/users_embeddings.pkl', 'wb') as f:
            pickle.dump({}, f)
        return {}

def save_user_embeddings(user_embeddings):
    with open('embeddings/users_embeddings.pkl', 'wb') as f:
        pickle.dump(user_embeddings, f)


def detect_faces_yunet(image):
    h, w = image.shape[:2]
    face_detector.setInputSize((w, h))
    faces = face_detector.detect(image)
    boxes = []
    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = face[:4]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            boxes.append([x1, y1, x2, y2])
    return boxes

def get_face_embedding(image):
    image = cv2.resize(image, (160, 160))
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        embedding = face_model(image_tensor).squeeze(0)
    return embedding

def is_real_face(face_image):
    resized = cv2.resize(face_image, (128, 128)).astype(np.float32)
    input_blob = resized.transpose(2, 0, 1)[np.newaxis, :] / 255.0
    ort_inputs = {spoof_model.get_inputs()[0].name: input_blob}
    output = spoof_model.run(None, ort_inputs)[0]
    prediction = np.argmax(output)
    return prediction == 1

def add_user_to_db(image, name, user_embeddings):
    boxes = detect_faces_yunet(image)
    if not boxes:
        st.warning("No face detected in the image.")
        return
    x1, y1, x2, y2 = boxes[0]
    face = image[y1:y2, x1:x2]
    if is_real_face(face):
        embedding = get_face_embedding(face)
        user_embeddings[name] = embedding.numpy()
        save_user_embeddings(user_embeddings)
        st.success(f"User {name} added successfully!")
    else:
        st.warning("Spoof face detected. Not adding to database.")

def recognize_faces(image, user_embeddings):
    boxes = detect_faces_yunet(image)
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
                if dist < min_dist and dist < 0.9:
                    min_dist = dist
                    name = user_name
            recognized.append((box, name))
    return recognized

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
        else:
            st.sidebar.warning("Please enter a name.")

st.sidebar.title("Webcam")
run_webcam = st.sidebar.button("Start Webcam")

if run_webcam:
    cap = cv2.VideoCapture(0)
    frame_display = st.empty()
    stop_button = st.sidebar.button("Stop Webcam")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognize_faces(frame_rgb, user_embeddings)
        for (x1, y1, x2, y2), name in results:
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_rgb, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        frame_display.image(frame_rgb, channels="RGB", use_column_width=True)
        if stop_button:
            break
    cap.release()
