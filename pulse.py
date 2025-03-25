import streamlit as st
import cv2
import numpy as np
import time
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot

class HeartRateMonitor:
    def __init__(self):
        self.detector = FaceDetector()
        self.plotter = LivePlot(640, 480, [20, 160], invert=True)
        self.buffer = []
        self.buffer_size = 10
        self.roi_size = (160, 120)
        self.gaussian_levels = 3
        self.amplification_factor = 170
        self.fps = 30
        self.time_buffer = []
        self.start_time = time.time()

    def process_frame(self, frame):
        img, bboxs = self.detector.findFaces(frame, draw=False)
        if bboxs:
            x, y, w, h = bboxs[0]['bbox']
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, self.roi_size)
            return roi, img
        return None, img

    def gaussian_pyramid(self, roi):
        pyramid = [roi]
        for _ in range(self.gaussian_levels):
            roi = cv2.pyrDown(roi)
            pyramid.append(roi)
        return pyramid

    def magnify_color(self, pyramid):
        magnified_pyramid = []
        for level in pyramid:
            level = level.astype(np.float32)
            magnified_pyramid.append(level)
        return magnified_pyramid

    def bandpass_filter(self, magnified_pyramid):
        filtered_pyramid = []
        for level in magnified_pyramid:
            fft = np.fft.fft2(level, axes=(0, 1))
            freqs = np.fft.fftfreq(level.shape[0])
            freqs_y = np.fft.fftfreq(level.shape[1])
            f = np.sqrt(freqs[:, None] ** 2 + freqs_y[None, :] ** 2)
            mask = (f >= 1 / self.fps) & (f <= 2 / self.fps)
            fft[~mask] = 0
            filtered = np.fft.ifft2(fft, axes=(0, 1)).real
            filtered_pyramid.append(filtered)
        return filtered_pyramid

    def calculate_bpm(self, filtered_pyramid):
        signal = filtered_pyramid[-1].mean(axis=(0, 1))
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(signal.size, 1 / self.fps)
        mask = (freqs >= 1) & (freqs <= 2)
        fft[~mask] = 0
        peak_freq = abs(freqs[np.argmax(abs(fft))])
        bpm = peak_freq * 60
        return bpm

def main():
    st.title("Real-Time Heart Rate Monitor")
    
    try:
        # Initialize camera - try different backends if needed
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:  # Try different backends
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                break
        
        if not cap.isOpened():
            st.error("""
                Could not access camera. Please:
                1. Ensure camera is connected
                2. Grant camera permissions
                3. Try refreshing the page
                """)
            return
            
        monitor = HeartRateMonitor()
        placeholder = st.empty()
        stop_button = st.button("Stop Monitoring")

        while True:
            if stop_button:
                st.write("Monitoring stopped")
                break
                
            success, frame = cap.read()
            if not success:
                st.warning("Could not read frame from camera")
                break
                
            roi, frame = monitor.process_frame(frame)
            if roi is not None:
                pyramid = monitor.gaussian_pyramid(roi)
                magnified_pyramid = monitor.magnify_color(pyramid)
                filtered_pyramid = monitor.bandpass_filter(magnified_pyramid)
                bpm = monitor.calculate_bpm(filtered_pyramid)

                monitor.buffer.append(bpm)
                if len(monitor.buffer) > monitor.buffer_size:
                    monitor.buffer.pop(0)

                if len(monitor.buffer) == monitor.buffer_size:
                    avg_bpm = np.mean(monitor.buffer)
                    cv2.putText(frame, f"BPM: {avg_bpm:.2f}", (frame.shape[1] // 2 - 50, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elapsed_time = time.time() - monitor.start_time
                    monitor.time_buffer.append(elapsed_time)
                    frame = monitor.plotter.update(avg_bpm, frame, monitor.time_buffer)
                else:
                    cv2.putText(frame, "Calculating BPM...", (frame.shape[1] // 2 - 100, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                placeholder.image(frame, channels="BGR")
            else:
                placeholder.image(frame, channels="BGR")
                st.warning("No face detected - please position your face in the frame")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    main()
