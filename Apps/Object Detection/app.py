import cv2
import datetime
import numpy as np
import streamlit as st
import pandas as pd
from ultralytics import YOLO

st.title("🎥 Real-Time Object Detection with YOLO + CSV Export")
st.sidebar.title("⚙️ Settings")

# Sidebar options
CONFIDENCE_THRESHOLD_LIMIT = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Choose source type
source_type = st.sidebar.radio("Choose Video Source", ("Upload Video", "Webcam"))

# Load YOLO model
model = YOLO("yolov8m.pt")

# ui state initialization
if "unique_objects" not in st.session_state:
    st.session_state.unique_objects = {}
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "video_ended" not in st.session_state:
    st.session_state.video_ended = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "last_detections" not in st.session_state:   
    st.session_state.last_detections = []

# yeh fast kyun nhi ho rha
def process_frame(frame, frame_count):
    detections = []

    # 3rd frane mai ru krne ke baad bhi nahi ho rha
    if frame_count % 5 == 0:
        start = datetime.datetime.now()
        results = model.track(frame, persist=True)[0]

        boxes = results.boxes
        bboxes = np.array(boxes.xyxy.cpu(), dtype="int")
        classes = np.array(boxes.cls.cpu(), dtype="int")
        confidence = np.array(boxes.conf.cpu(), dtype="float")
        ids = np.array(boxes.id.cpu(), dtype="int") if boxes.id is not None else []

        for cls, bbox, conf, obj_id in zip(classes, bboxes, confidence, ids):
            if conf < CONFIDENCE_THRESHOLD_LIMIT:
                continue
            detections.append((cls, bbox, conf, obj_id))

        # time waste nahi ho ga
        st.session_state.last_detections = detections

        # FPS calculation
        end = datetime.datetime.now()
        total = (end - start).total_seconds()
        st.session_state.fps = f"FPS: {1/total:.2f}" if total > 0 else "FPS: --"
    else:
        detections = st.session_state.last_detections

    # Always show FPS (no flickering)
    if "fps" in st.session_state:
        cv2.putText(frame, st.session_state.fps, (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # green box
    for cls, bbox, conf, obj_id in detections:
        object_name = model.names[cls]
        if object_name not in st.session_state.unique_objects:
            st.session_state.unique_objects[object_name] = set()
        st.session_state.unique_objects[object_name].add(int(obj_id))

        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (37, 245, 75), 1)
        cv2.putText(
            frame,
            f"{object_name} ID:{obj_id} {conf*100:.0f}%",
            (x, y - 5),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (37, 245, 75),
            1,
        )

    return frame

# ---- Video Source ----
cap = None
if source_type == "Upload Video":
    video_source = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if video_source is not None:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(video_source.read())
        tfile.close()
        cap = cv2.VideoCapture("temp_video.mp4")
    else:
        st.info("👆 Please upload a video file to start detection.")

elif source_type == "Webcam":
    cap = cv2.VideoCapture(0)

# ---- Display ----
stframe = st.empty()

# Play / Pause Buttons
btn_col1, btn_col2, btn_col3 = st.columns([4, 1, 4])
with btn_col2:
    play = st.button("▶️ Play")
    pause = st.button("⏸️ Pause")

if play:
    st.session_state.is_playing = True
    st.session_state.video_ended = False
if pause:
    st.session_state.is_playing = False

# ---- Processing Loop ----
frame_count = 0
if cap is not None and cap.isOpened():
    while st.session_state.is_playing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.session_state.is_playing = False
            st.session_state.video_ended = True
            break

        frame_count += 1
        frame = process_frame(frame, frame_count)   
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.last_frame = frame
        stframe.image(frame, channels="RGB")

    # Keep last frame visible
    if st.session_state.last_frame is not None:
        stframe.image(st.session_state.last_frame, channels="RGB")

    if not st.session_state.is_playing and not st.session_state.video_ended:
        st.info("⏸️ Video paused.")

    if st.session_state.video_ended:
        st.success("✅ Video ended.")

    if not st.session_state.is_playing and st.session_state.unique_objects:
        # ---- Final Object Summary ----
        final_counts = {obj: len(ids) for obj, ids in st.session_state.unique_objects.items()}
        df = pd.DataFrame(list(final_counts.items()), columns=["Object", "Count"])

        st.subheader("📊 Final Object Counts")
        st.table(df)

        # CSV Export Button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="object_counts.csv",
            mime="text/csv"
        )
else:
    st.info("⚠️ No video source available.")

# ---- Cleanup ----
def cleanup():
    if cap is not None:
        cap.release()

import atexit
atexit.register(cleanup)
