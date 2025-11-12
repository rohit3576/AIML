# 🎥 Abnormal Behavior Detector (Streamlit App)

This is a Streamlit application designed to detect and flag abnormal human behavior from a video stream (file or webcam). It fuses multiple computer vision models to identify anomalies based on skeleton movement, facial emotion, and drowsiness.

The app is built to **show and log only the frames where an abnormality is detected**.

## ✨ Key Features

* **Multi-Backend Support:** Choose between:
    * **YOLOv8-Pose:** For multi-person pose estimation and tracking.
    * **MediaPipe Pose:** For fast, single-person pose estimation.
* **Multi-Modal Anomaly Detection:**
    * **🏃 Skeleton Movement:** Detects sudden, erratic, or high-velocity wrist and torso movements using Median Absolute Deviation (MAD) on joint velocities.
    * **😠 Facial Emotion:** Uses the `fer` library to detect negative emotions (e.g., angry, fear, disgust, sad).
    * **😴 Drowsiness (EAR):** Uses MediaPipe FaceMesh to calculate the Eye Aspect Ratio (EAR) and flags prolonged eye closure.
* **Input Sources:** Works with both pre-recorded video files (`.mp4`, `.mov`, etc.) and live webcam feeds.
* **Event Logging:** Automatically logs all detected abnormal events with timestamps and reasons (e.g., "Movement", "Drowsy") to `logs/activity_log.csv`.
* **Interactive UI:** A simple Streamlit sidebar allows you to configure the input source, backend model, active detection modules, and sensitivity thresholds.



## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    All required packages are listed in the `requirements.txt` file. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `torch` and `torchvision` are large libraries. Installation may take a few minutes.)*

## 🚀 How to Run

1.  From your terminal (with the virtual environment activated), run the Streamlit app:
    ```bash
    streamlit run streamlit_abnormal_detector_fusion.py
    ```

2.  Streamlit will open the application in your default web browser (usually at `http://localhost:8501`).

3.  **In the sidebar:**
    * Choose your **Input Source** (Webcam or Upload Video).
    * Select the **Skeleton Backend** (YOLOv8 is recommended for multiple people).
    * Enable or disable detection **Modules** (Movement, Emotion, Drowsiness).
    * Adjust **Thresholds** as needed.

4.  Click the **▶️ Start** button to begin processing.

## 📁 Logging

* All detected abnormal events are saved in the `logs/` directory.
* `activity_log.csv`: A CSV file for easy data analysis (can be downloaded from the sidebar).
* `activity_log.txt`: A human-readable text log.