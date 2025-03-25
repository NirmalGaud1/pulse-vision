import streamlit as st
import cv2
import numpy as np
import time
from cvzone.FaceDetectionModule import FaceDetector
from collections import deque

# Constants
BUFFER_SIZE = 30  # For smoothing readings
MIN_FACE_CONFIDENCE = 0.8

# Initialize once and cache
@st.cache_resource
def get_face_detector():
    return FaceDetector(minDetectionCon=0.5)

def main():
    st.title("Advanced Health Parameter Monitor")
    
    # Layout columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("Vital Signs")
        bpm_placeholder = st.empty()
        bp_placeholder = st.empty()
        spo2_placeholder = st.empty()
        stress_placeholder = st.empty()
        status_placeholder = st.empty()
    
    # Upload video
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    if not uploaded_file:
        st.warning("Please upload a video file")
        st.stop()
    
    # Save to temp file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize video capture
    cap = cv2.VideoCapture("temp_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = get_face_detector()
    
    # Buffers for smoothing
    bpm_buffer = deque(maxlen=BUFFER_SIZE)
    bp_buffer = deque(maxlen=BUFFER_SIZE)
    spo2_buffer = deque(maxlen=BUFFER_SIZE)
    
    # Processing parameters
    frame_skip = 2  # Process every nth frame to increase speed
    frame_count = 0
    processing_start = time.time()
    
    stop_button = st.button("Stop Processing")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames for faster processing
        
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Face detection
        frame, bboxs = detector.findFaces(frame)
        
        if bboxs and bboxs[0]['score'] > MIN_FACE_CONFIDENCE:
            # Get face ROI
            x, y, w, h = bboxs[0]['bbox']
            face_roi = frame[y:y+h, x:x+w]
            
            # Placeholder for actual signal processing
            # In a real app, you would extract PPG signal here
            
            # Simulate health parameters (replace with actual calculations)
            current_bpm = np.clip(np.random.normal(72, 5), 60, 100)
            current_bp = f"{np.random.randint(110, 130)}/{np.random.randint(70, 85)}"
            current_spo2 = np.random.randint(95, 100)
            current_stress = np.random.randint(0, 50)
            
            # Update buffers
            bpm_buffer.append(current_bpm)
            bp_buffer.append(current_bp)
            spo2_buffer.append(current_spo2)
            
            # Calculate smoothed values
            avg_bpm = np.mean(bpm_buffer) if bpm_buffer else 0
            avg_bp = max(set(bp_buffer), key=bp_buffer.count) if bp_buffer else "0/0"
            avg_spo2 = np.mean(spo2_buffer) if spo2_buffer else 0
            
            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"BPM: {avg_bpm:.1f}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update dashboard
            with col2:
                bpm_placeholder.metric("Heart Rate", f"{avg_bpm:.1f} bpm", "±2")
                bp_placeholder.metric("Blood Pressure", avg_bp, "±5")
                spo2_placeholder.metric("SpO2", f"{avg_spo2:.0f}%", "±1")
                stress_placeholder.metric("Stress Level", f"{current_stress}%", "±5")
                status_placeholder.success("Normal reading")
        else:
            with col2:
                status_placeholder.warning("No face detected")
        
        # Display processing speed
        processing_time = time.time() - processing_start
        fps_actual = frame_count / processing_time if processing_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps_actual:.1f}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display frame
        video_placeholder.image(frame, channels="BGR")
    
    cap.release()
    cv2.destroyAllWindows()
    st.success("Video processing completed")

if __name__ == "__main__":
    main()
