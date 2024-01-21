import tensorflow as tf
import torch
import torchvision.transforms as T
import torch.nn.functional as F

from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import streamlit as st
import threading
from typing import Union
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225


def classify_transforms(size=64):
    return T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


cascade = cv2.CascadeClassifier("burstman/python_tuto/master/face_emotion_project/haarcascade_frontalface_default.xml")


emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to load VGG16 model

# Load the saved model


def load_VGG_16_model():
    model = tf.keras.models.load_model('burstman/python_tuto/master/face_emotion_project/emotion_model.keras', compile=False)  # type: ignore
    print(type(model))
    return model


model_VGG = load_VGG_16_model()


def load_yolo_model():
    yolo_model = torch.hub.load(
        'ultralytics/yolov5', 'custom', path='burstman/python_tuto/master/face_emotion_project/best.pt')
    print(type(yolo_model))
    return yolo_model


model_YOLO = load_yolo_model()

reset_button = st.button("Reset Viewport")

class VideoTransformer_VGG():
    # transform() is running in another thread, then a lock object is used here for thread-safety.
    frame_lock: threading.Lock
    in_image: Union[np.ndarray, None]

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.in_image = None
        self.scale = None
        self.minNeighbors = None
        self.RectColor = None

    def recv(self, frame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, self.scale, self.minNeighbors)  # type: ignore
        self.vgg16_model_input_size = (48, 48)
        self.yolov5_model_input_size = (224, 224)

        for x, y, w, h in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, self.vgg16_model_input_size)
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            face = np.reshape(face, [1, 48, 48, 3])/255.0

            # Assuming model_VGG is your VGG model
            predict = model_VGG.predict(face)
            prediction_index = np.argmax(predict, axis=-1)[0]
            prediction_emotion = emotions[prediction_index]

            # Get top 5 predictions
            top5_indices = np.argsort(predict[0])[::-1][:5]
            top5_emotions = [emotions[i] for i in top5_indices]
            top5_probs = predict[0][top5_indices]

            # Draw rectangle and put predicted emotion
            cv2.rectangle(frm, (x, y), (x+w, y+h), self.RectColor, 2) # type: ignore
            cv2.putText(frm, prediction_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.RectColor, 2) # type: ignore

            # Display top 5 emotions and their probabilities
            column_x, column_y = 10, 30
            for emotion, prob in zip(top5_emotions, top5_probs):
                text = f'{prob:.2f} {emotion}'
                cv2.putText(frm, text, (column_x, column_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.RectColor, 2) # type: ignore
                column_y += 25  # Adjust this value based on the desired vertical spacing between lines
            column_x += 150  # Adjust this value based on the desired horizontal spacing between columns
            column_y = 100
        return av.VideoFrame.from_ndarray(frm, format='bgr24')


class VideoTransformer_YOLO():
    # transform() is running in another thread, then a lock object is used here for thread-safety.
    frame_lock: threading.Lock
    in_image: Union[np.ndarray, None]

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.in_image = None
        self.scale = None
        self.minNeighbors = None
        self.RectColor = None
        self.yolo_result_color=None
    def recv(self, frame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, self.scale, self.minNeighbors)  # type: ignore
        self.vgg16_model_input_size = (48, 48)
        

        for x, y, w, h in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, self.vgg16_model_input_size)
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            transformations = classify_transforms()
            convert_tensor = transformations(face)
            convert_tensor = convert_tensor.unsqueeze(0)
            convert_tensor = convert_tensor.to(device)
            results = model_YOLO(convert_tensor)
            pred = F.softmax(results, dim=1)
            max_value, max_index = torch.max(pred, dim=1)
            prediction_emotion = model_YOLO.names[max_index.item()]
            cv2.putText(frm, prediction_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.RectColor, 2)  # type: ignore
            
            cv2.rectangle(frm, (x, y), (x+w, y+h),
                          self.RectColor, 2)  # type: ignore
            column_x, column_y = 10, 30

            for i, prob in enumerate(pred):
                top5i = prob.argsort(0, descending=True)[:5].tolist()
                for j in top5i:
                    text = f'{prob[j]:.2f} {model_YOLO.names[j]}'
                    cv2.putText(frm, text, (column_x, column_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.RectColor, 2) # type: ignore
                    column_y += 25  # Adjust this value based on the desired vertical spacing between lines
                column_x += 150  # Adjust this value based on the desired horizontal spacing between columns
                column_y = 100             
            
        return av.VideoFrame.from_ndarray(frm, format='bgr24')


# Function to create a Streamlit session for VGG16
def create_session_VGG():
    st.title("Face Emotion Detection VGG16") 
    ctx = webrtc_streamer(key="VGG16", video_processor_factory=VideoTransformer_VGG, rtc_configuration=RTCConfiguration( # type: ignore
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))  # type: ignore
    return ctx
# Function to create a Streamlit session for YOLOv5
def create_session_YOLO():
    st.title("Face Emotion Detection YOLOv5")
    ctx = webrtc_streamer(key="YOLO", video_processor_factory=VideoTransformer_YOLO, # type: ignore
                               rtc_configuration=RTCConfiguration(
                                   {"iceServers": [
                                       {"urls": ["stun:stun.l.google.com:19302"]}]}
                               )
                               ) # type: ignore
    return ctx


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    # Reversing the order to convert from RGB to BGR
    bgr_color = tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))
    return bgr_color
# Create a button to reset the viewport


# Create separate sessions for VGG16 and YOLOv5
ctx_VGG = create_session_VGG()
ctx_YOLO = create_session_YOLO()

if ctx_VGG.video_transformer:
    if st.button("snapshot"):
        with ctx.video_transformer.frame_lock:  # type: ignore
            frame = ctx.video_transformer.in_image  # type: ignore

        if frame is not None:
            st.write("Input image:")
            st.image(frame, channels="BGR")
            cv2.imwrite('face_detection/frame_on_button_click.png', frame)

        else:
            st.warning("No frames available yet.")
    color = st.color_picker('Pick A Color', '#df4cc5')
    rgb = hex_to_bgr(color)
    ctx_VGG.video_transformer.RectColor = rgb
    st.write(rgb)
    values_slide = st.slider('Select the scale value', 1.0, 1.8, 1.1)
    ctx_VGG.video_transformer.scale = values_slide
    values_min_neighboor = st.slider('Select the minNeighbors value', 1, 10, 5)
    ctx_VGG.video_transformer.minNeighbors = values_min_neighboor
        # Check if the reset button is pressed
    if reset_button:
        st.caching.clear_cache() # type: ignore
        st.experimental_rerun()


if ctx_YOLO.video_transformer:
    if st.button("snapshot"):
        with ctx.video_transformer.frame_lock:  # type: ignore
            frame = ctx.video_transformer.in_image  # type: ignore

        if frame is not None:
            st.write("Input image:")
            st.image(frame, channels="BGR")
            cv2.imwrite('face_detection/frame_on_button_click.png', frame)

        else:
            st.warning("No frames available yet.")
    color = st.color_picker('Pick A Color', '#df4cc5')
    rgb = hex_to_bgr(color)
    ctx_YOLO.video_transformer.RectColor = rgb
    st.write(rgb)
    values_slide = st.slider('Select the scale value', 1.0, 1.8, 1.1)
    ctx_YOLO.video_transformer.scale = values_slide
    values_min_neighboor = st.slider('Select the minNeighbors value', 1, 10, 5)
    ctx_YOLO.video_transformer.minNeighbors = values_min_neighboor
    if reset_button:
        st.caching.clear_cache() # type: ignore
        st.experimental_rerun()



