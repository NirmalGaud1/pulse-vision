import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import time
import tensorflow as tf
from tensorflow.keras import layers, models

# Page config
st.set_page_config(
    page_title="PulseVision - Real-time Heart Rate Monitor",
    page_icon="❤️",
    layout="wide"
)

# Main UI
st.title("❤️ PulseVision - Real-time Heart Rate Monitoring")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Captured Image")
    image_placeholder = st.empty()

with col2:
    st.subheader("Vital Signs")
    bpm_placeholder = st.empty()
    bp_placeholder = st.empty()
    signal_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Controls")
    capture_button = st.button("Capture Image")
    stop_button = st.button("Stop Monitoring")

# State management
if 'running' not in st.session_state:
    st.session_state.running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Parameters
BUFFER_SIZE = 150
bpm_buffer = []
last_bpm = 0
last_bp = "120/80"
fps = 30
update_interval = 1
signal_history = []
frame_buffer = []

# Initialize model
def build_attention_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x_motion = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x_motion = layers.Conv2D(32, (3, 3), activation='relu')(x_motion)
    x_app = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x_app = layers.Conv2D(32, (3, 3), activation='relu')(x_app)
    attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x_app)
    attention = layers.Multiply()([x_motion, attention])
    x = layers.AveragePooling2D((2, 2))(attention)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=output)
    return model

# Bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.75, highcut=3.0, fs=30, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Process ROI
def process_roi(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    a_channel = cv2.resize(a_channel, (160, 120))
    processed = (a_channel - a_channel.mean()) / (a_channel.std() + 1e-6)
    if len(frame_buffer) > 0:
        prev_frame = frame_buffer[-1]
        diff = processed - prev_frame
        processed = processed * 0.7 + diff * 0.3
    frame_buffer.append(processed.copy())
    if len(frame_buffer) > 5:
        frame_buffer.pop(0)
    return processed

def estimate_blood_pressure(bpm):
    systolic = 120 + (bpm - 60) * 0.2
    diastolic = 80 + (bpm - 60) * 0.1
    return f"{int(systolic)}/{int(diastolic)}"

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if 'model' not in st.session_state:
    st.session_state.model = build_attention_model((120, 160, 1))

if capture_button:
    try:
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
        ret, frame = st.session_state.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                x1, y1, x2, y2 = x, y, x + w, y + h
                roi = frame[y1:y1 + (y2 - y1) // 3, x1:x2]
                processed = process_roi(roi)
                signal = processed.mean()
                signal_history.append(signal)

                if len(signal_history) > fps:
                    filtered = bandpass_filter(signal_history[-fps * 3:])
                    peaks, _ = find_peaks(filtered, distance=fps // 2)
                    if len(peaks) > 1:
                        intervals = np.diff(peaks) / fps
                        bpm = 60 / np.mean(intervals)
                        bpm_buffer.append(bpm)
                        if len(bpm_buffer) > BUFFER_SIZE:
                            bpm_buffer.pop(0)
                        last_bpm = np.median(bpm_buffer)
                        last_bp = estimate_blood_pressure(last_bpm)
                        bpm_placeholder.metric("Current Heart Rate", f"{int(last_bpm)} BPM" if last_bpm > 0 else "---")
                        bp_placeholder.metric("Estimated Blood Pressure", last_bp)

                signal_img = np.zeros((200, 400, 3), dtype=np.uint8)
                if len(signal_history) > 10:
                    signals = signal_history[-100:]
                    normalized = (signals - np.min(signals)) / (np.max(signals) - np.min(signals) + 1e-6)
                    for i in range(1, len(normalized)):
                        cv2.line(signal_img, (int((i - 1) * 4), int(180 * (1 - normalized[i - 1]))),
                                 (int(i * 4), int(180 * (1 - normalized[i]))), (0, 255, 0), 2)
                signal_placeholder.image(signal_img, caption="Pulse Signal", use_container_width=True)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(frame, channels="RGB", use_container_width=True)
            else:
                st.warning("No face detected.")
        else:
            st.error("Failed to capture image.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if stop_button:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.stop()
