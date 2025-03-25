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
    st.subheader("Live Camera Feed")
    camera_placeholder = st.empty()

with col2:
    st.subheader("Vital Signs")
    bpm_placeholder = st.empty()
    bp_placeholder = st.empty()
    signal_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Controls")
    start_button = st.button("Start Monitoring")
    stop_button = st.button("Stop Monitoring")

# State management
if 'running' not in st.session_state:
    st.session_state.running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Parameters
BUFFER_SIZE = 150  # Larger buffer for more stable readings
bpm_buffer = []
last_bpm = 0
last_bp = "120/80"
fps = 30
update_interval = 1  # Update BPM every second
last_update_time = time.time()
signal_history = []
time_history = []
frame_buffer = []

# Initialize a simple attention-based model (inspired by MTTS-CAN)
def build_attention_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Motion branch (temporal features)
    x_motion = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x_motion = layers.Conv2D(32, (3, 3), activation='relu')(x_motion)
    
    # Appearance branch (spatial features)
    x_app = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x_app = layers.Conv2D(32, (3, 3), activation='relu')(x_app)
    
    # Attention mechanism
    attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x_app)
    attention = layers.Multiply()([x_motion, attention])
    
    # Continue processing
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

# Process ROI with attention mechanism
def process_roi(roi):
    # Convert to LAB color space
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    
    # Apply attention-inspired processing
    a_channel = cv2.resize(a_channel, (160, 120))
    processed = (a_channel - a_channel.mean()) / (a_channel.std() + 1e-6)
    
    # Apply temporal smoothing (similar to MTTS-CAN)
    if len(frame_buffer) > 0:
        prev_frame = frame_buffer[-1]
        diff = processed - prev_frame
        processed = processed * 0.7 + diff * 0.3
    
    frame_buffer.append(processed.copy())
    if len(frame_buffer) > 5:  # Keep a small buffer
        frame_buffer.pop(0)
    
    return processed

def estimate_blood_pressure(bpm):
    # Simple estimation based on heart rate
    systolic = 120 + (bpm - 60) * 0.2
    diastolic = 80 + (bpm - 60) * 0.1
    return f"{int(systolic)}/{int(diastolic)}"

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if start_button and not st.session_state.running:
    st.session_state.running = True
    # Try multiple camera indices
    for i in range(3):  # Try 0, 1, 2
        st.session_state.cap = cv2.VideoCapture(i)
        if st.session_state.cap.isOpened():
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            signal_history = []
            time_history = []
            start_time = time.time()
            break
    
    if not st.session_state.cap or not st.session_state.cap.isOpened():
        st.error("Could not open any camera. Please check your camera connection.")
        st.session_state.running = False
        st.session_state.cap = None
    else:
        # Initialize model
        st.session_state.model = build_attention_model((120, 160, 1))

if stop_button and st.session_state.running:
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

# Main processing loop
while st.session_state.running and st.session_state.cap is not None:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("Failed to capture frame")
        st.session_state.running = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        break
    
    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Extract ROI (forehead region)
        roi = frame[y1:y1 + (y2 - y1) // 3, x1:x2]

        # Process signal with attention-inspired approach
        processed = process_roi(roi)
        signal = processed.mean()
        signal_history.append(signal)
        time_history.append(time.time() - start_time)

        # Update BPM periodically
        current_time = time.time()
        if (current_time - last_update_time > update_interval) and (len(signal_history) > fps):
            # Apply bandpass filter
            filtered = bandpass_filter(signal_history[-fps*3:])  # Use 3 seconds of data
            
            # Find peaks in the filtered signal
            peaks, _ = find_peaks(filtered, distance=fps//2)
            
            if len(peaks) > 1:
                # Calculate BPM from peak intervals
                intervals = np.diff(peaks) / fps
                bpm = 60 / np.mean(intervals)
                
                # Smooth the BPM reading
                bpm_buffer.append(bpm)
                if len(bpm_buffer) > BUFFER_SIZE:
                    bpm_buffer.pop(0)
                
                last_bpm = np.median(bpm_buffer)  # Use median for robustness
                last_bp = estimate_blood_pressure(last_bpm)
                last_update_time = current_time

        # Draw face rectangle and BPM text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"BPM: {int(last_bpm) if last_bpm > 0 else 'Calculating...'}",
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display pulse signal
    signal_img = np.zeros((200, 400, 3), dtype=np.uint8)
    if len(signal_history) > 10:
        # Normalize and plot the signal
        signals = signal_history[-100:]
        normalized = (signals - np.min(signals)) / (np.max(signals) - np.min(signals) + 1e-6)
        
        for i in range(1, len(normalized)):
            cv2.line(
                signal_img,
                (int((i-1)*4), int(180*(1-normalized[i-1]))),
                (int(i*4), int(180*(1-normalized[i]))),
                (0, 255, 0), 2
            )

    # Display everything
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_placeholder.image(frame, channels="RGB", use_container_width=True)
    bpm_placeholder.metric("Current Heart Rate", f"{int(last_bpm)} BPM" if last_bpm > 0 else "---")
    bp_placeholder.metric("Estimated Blood Pressure", last_bp)
    signal_placeholder.image(signal_img, caption="Pulse Signal", use_container_width=True)

    # Small delay to prevent freezing
    time.sleep(0.03)

# Cleanup
if st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None

# Instructions
st.sidebar.markdown("""
### How to Use:
1. Click **Start Monitoring**
2. Position face in camera view
3. Remain still for accurate readings
4. Click **Stop Monitoring** when done

### Tips:
- Ensure good, even lighting
- Keep your face still during measurement
- Position your forehead clearly in view
- Avoid strong backlight or shadows

### Technical Details:
- Uses attention-inspired signal processing
- Combines spatial and temporal features
- Bandpass filtered (0.75-3.0 Hz) for robust readings
- Median filtering for stable BPM output
""")
