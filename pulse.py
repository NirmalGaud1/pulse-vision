import streamlit as st
import cv2
import numpy as np
import time
from cvzone.FaceDetectionModule import FaceDetector
from collections import deque

# Constants
BUFFER_SIZE = 30
MIN_FACE_CONFIDENCE = 0.5
TARGET_FPS = 25  # Target frames per second for display

@st.cache_resource
def get_face_detector():
    return FaceDetector(minDetectionCon=MIN_FACE_CONFIDENCE)

def main():
    st.title("🫀 Real-Time Health Monitoring Dashboard")
    
    # Layout columns
    col_video, col_stats = st.columns([3, 1])
    
    with col_video:
        st.subheader("Live Video Analysis")
        video_placeholder = st.empty()
        fps_display = st.empty()
    
    with col_stats:
        st.subheader("Vital Signs")
        bpm_placeholder = st.empty()
        bp_placeholder = st.empty()
        spo2_placeholder = st.empty()
        stress_placeholder = st.empty()
        status_placeholder = st.empty()
        proc_time = st.empty()
    
    # Upload video
    uploaded_file = st.file_uploader("📤 Upload video file", type=['mp4', 'mov', 'avi'])
    if not uploaded_file:
        st.info("Please upload a video file to begin analysis")
        st.stop()
    
    # Save to temp file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize video with progress bar
    cap = cv2.VideoCapture("temp_video.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_slider = st.slider("Video Position", 0, total_frames-1, 0)
    
    detector = get_face_detector()
    
    # Buffers for smoothing
    bpm_buffer = deque(maxlen=BUFFER_SIZE)
    bp_buffer = deque(maxlen=BUFFER_SIZE)
    spo2_buffer = deque(maxlen=BUFFER_SIZE)
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    last_frame_time = start_time
    
    # Playback control
    play_button = st.button("▶️ Play/Pause")
    stop_button = st.button("⏹ Stop")
    
    # Main processing loop
    while cap.isOpened() and not stop_button:
        if play_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Control playback speed
            elapsed = current_time - last_frame_time
            target_delay = 1/TARGET_FPS
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)
            
            last_frame_time = time.time()
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            frame_slider = frame_count - 1
            
            # Resize for display while maintaining aspect
            display_frame = cv2.resize(frame, (640, 360))
            
            # Face detection
            processed_frame, bboxs = detector.findFaces(display_frame.copy())
            
            if bboxs:
                main_face = max(bboxs, key=lambda x: x['score'][0])
                confidence = main_face['score'][0]
                
                if confidence > MIN_FACE_CONFIDENCE:
                    x, y, w, h = main_face['bbox']
                    
                    # Draw dynamic face tracking box
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(processed_frame, (x+w//2, y+h//2), 5, (0, 0, 255), -1)
                    
                    # Simulate health parameters (replace with real processing)
                    current_bpm = np.clip(72 + 10*np.sin(frame_count/10), 60, 100)
                    current_bp = f"{int(120 + 5*np.sin(frame_count/15))}/{int(80 + 5*np.cos(frame_count/20))}"
                    current_spo2 = np.clip(98 - 2*np.sin(frame_count/25), 95, 100)
                    current_stress = np.clip(20 + 15*np.sin(frame_count/30), 0, 50)
                    
                    # Update buffers
                    bpm_buffer.append(current_bpm)
                    bp_buffer.append(current_bp)
                    spo2_buffer.append(current_spo2)
                    
                    # Calculate smoothed values
                    avg_bpm = np.mean(bpm_buffer) if bpm_buffer else 0
                    avg_bp = max(set(bp_buffer), key=bp_buffer.count) if bp_buffer else "0/0"
                    avg_spo2 = np.mean(spo2_buffer) if spo2_buffer else 0
                    
                    # Add vital signs overlay
                    cv2.putText(processed_frame, f"❤️ {avg_bpm:.1f} BPM", (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Update dashboard
                    with col_stats:
                        bpm_placeholder.metric("Heart Rate", f"{avg_bpm:.1f} bpm", 
                                             f"{'↑' if current_bpm > avg_bpm else '↓'} {abs(current_bpm-avg_bpm):.1f}")
                        bp_placeholder.metric("Blood Pressure", avg_bp, "±5")
                        spo2_placeholder.metric("Oxygen", f"{avg_spo2:.0f}%", "±1")
                        stress_placeholder.metric("Stress", f"{current_stress:.0f}%", 
                                               f"{'↑' if current_stress > 30 else '↓'}")
                        status_placeholder.success(f"✅ Tracking (Conf: {confidence:.2f})")
                else:
                    with col_stats:
                        status_placeholder.warning(f"⚠️ Low confidence: {confidence:.2f}")
            else:
                with col_stats:
                    status_placeholder.error("❌ No face detected")
            
            # Calculate actual FPS
            processing_time = time.time() - start_time
            actual_fps = frame_count / processing_time if processing_time > 0 else 0
            
            # Add FPS counter to frame
            cv2.putText(processed_frame, f"FPS: {actual_fps:.1f}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display the processed frame with animations
            video_placeholder.image(processed_frame, channels="BGR")
            fps_display.caption(f"Frame: {frame_count}/{total_frames} | FPS: {actual_fps:.1f}")
            proc_time.caption(f"Processing time: {time.time() - current_time:.3f}s")
    
    cap.release()
    st.success("✅ Analysis complete!")
    st.balloons()

if __name__ == "__main__":
    main()
