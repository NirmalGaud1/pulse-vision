import streamlit as st
import cv2
import numpy as np
import time
from cvzone.FaceDetectionModule import FaceDetector

# Initialize once and cache
@st.cache_resource
def get_face_detector():
    return FaceDetector()

def main():
    st.title("Real-Time Heart Rate Monitor")
    
    # Camera selection
    camera_option = st.radio("Select Camera Source:", 
                           ["Webcam", "Upload Video"], 
                           horizontal=True)
    
    if camera_option == "Webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("""
            Could not access camera. Please:
            1. Ensure no other app is using the camera
            2. Refresh the page and allow camera permissions
            3. Try a different browser (Chrome works best)
            """)
            return
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov'])
        if not uploaded_file:
            st.stop()
        # Save to temp file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        cap = cv2.VideoCapture("temp_video.mp4")

    detector = get_face_detector()
    placeholder = st.empty()
    stop_button = st.button("Stop Monitoring")

    while True:
        if stop_button:
            st.success("Monitoring stopped")
            break
            
        success, frame = cap.read()
        if not success:
            st.warning("Video ended or frame not available")
            break
            
        # Face detection and processing
        frame, bboxs = detector.findFaces(frame)
        
        if bboxs:
            x, y, w, h = bboxs[0]['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Placeholder for your heart rate calculation logic
            bpm = np.random.randint(60, 100)  # Replace with actual calculation
            cv2.putText(frame, f"Estimated BPM: {bpm}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Face Detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame
        placeholder.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
