import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, lfilter
import time
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="PulseVision - Real-time Heart Rate Monitor",
    page_icon="❤️",
    layout="wide"
)

# Sidebar
st.sidebar.title("Settings")
model_choice = st.sidebar.radio(
    "Select Face Detector",
    ("YOLOv8 (Recommended)", "OpenCV Haar Cascade")
)

# Initialize YOLOv8 (load only once)
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n-face.pt')

if model_choice == "YOLOv8 (Recommended)":
    model = load_yolo()

# Main UI
st.title("❤️ PulseVision - Real-time Heart Rate Monitoring")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("Vital Signs")
    bpm_placeholder = st.empty()
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

# Parameters
BUFFER_SIZE = 10
bpm_buffer = []
last_bpm = 0
fps = 30
update_interval = 1  # Update BPM every second
last_update_time = time.time()
signal_history = []
time_history = []

# Bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=1.0, highcut=2.0, fs=30, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Process ROI
def process_roi(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a_channel = lab[:,:,1]
    a_channel = cv2.resize(a_channel, (160, 120))
    processed = (a_channel - a_channel.mean()) / a_channel.std()
    return processed

# Start/Stop handlers
if start_button:
    st.session_state.running = True
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.cap.set(3, 640)
    st.session_state.cap.set(4, 480)
    signal_history = []
    time_history = []
    start_time = time.time()

if stop_button:
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()

# Main processing loop
while st.session_state.running:
    if st.session_state.cap is None:
        break
        
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("Failed to capture video")
        st.session_state.running = False
        break
    
    # Face detection
    if model_choice == "YOLOv8 (Recommended)":
        results = model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        boxes = face_cascade.detectMultiScale(gray, 1.1, 4)
        boxes = [[x, y, x+w, y+h] for (x, y, w, h) in boxes]
    
    if len(boxes) > 0:
        # Get first face
        if model_choice == "YOLOv8 (Recommended)":
            x1, y1, x2, y2 = boxes[0].astype(int)
        else:
            x1, y1, x2, y2 = boxes[0]
        
        # Extract ROI (forehead)
        roi = frame[y1:y1 + (y2-y1)//3, x1:x2]
        
        # Process signal
        processed = process_roi(roi)
        signal = processed.mean()
        signal_history.append(signal)
        time_history.append(time.time() - start_time)
        
        # Update BPM
        if (time.time() - last_update_time > update_interval) and (len(signal_history) > fps):
            filtered = bandpass_filter(signal_history[-fps:])
            fft = np.fft.rfft(filtered)
            freqs = np.fft.rfftfreq(len(filtered), d=1.0/fps)
            
            mask = (freqs >= 1.0) & (freqs <= 2.0)
            if np.any(mask):
                peak_freq = freqs[mask][np.argmax(np.abs(fft[mask]))]
                bpm = peak_freq * 60
                bpm_buffer.append(bpm)
                if len(bpm_buffer) > BUFFER_SIZE:
                    bpm_buffer.pop(0)
                last_bpm = np.mean(bpm_buffer)
                last_update_time = time.time()
        
        # Visualization
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"BPM: {int(last_bpm) if last_bpm > 0 else 'Calculating...'}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display signal
        signal_img = np.zeros((200, 400, 3), dtype=np.uint8)
        if len(signal_history) > 10:
            normalized_signals = (signal_history[-100:] - np.min(signal_history[-100:])) / \
                               (np.max(signal_history[-100:]) - np.min(signal_history[-100:]) + 1e-6)
            for i in range(1, len(normalized_signals)):
                cv2.line(
                    signal_img, 
                    (int((i-1)*4), int(150*(1-normalized_signals[i-1]))), 
                    (int(i*4), int(150*(1-normalized_signals[i])))), 
                    (0, 255, 0), 2
                )
    
    # Convert to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update displays
    video_placeholder.image(frame, channels="RGB", use_column_width=True)
    bpm_placeholder.metric("Current Heart Rate", f"{int(last_bpm)} BPM" if last_bpm > 0 else "---")
    
    if 'signal_img' in locals():
        signal_placeholder.image(signal_img, caption="Pulse Signal", use_column_width=True)
    
    # Small delay to prevent high CPU usage
    time.sleep(0.03)

# Cleanup
if not st.session_state.running and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None

# Instructions
st.sidebar.markdown("""
### How to Use:
1. Select face detector (YOLOv8 recommended)
2. Click **Start Monitoring**
3. Position face in camera view
4. Remain still for accurate readings
5. Click **Stop Monitoring** when done

### Tips:
- Ensure good lighting
- Minimize head movements
- Avoid strong backlight
""")
