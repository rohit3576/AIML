# 🎥 Real-Time Object Detection with YOLO + CSV Export

This is a Streamlit application that performs real-time object detection and tracking using the YOLOv8 model. It allows you to process a video file or webcam feed, displays the results, and provides a summary of all unique objects detected, which can be exported as a CSV file.

## ✨ Key Features

* **YOLOv8 Powered:** Uses the powerful `yolov8m.pt` model for fast and accurate detection.
* **Object Tracking:** Assigns a unique ID to each detected object and tracks it across frames.
* **Dual Source:** Supports both **video file uploads** (`.mp4`, `.avi`, etc.) and **live webcam** feeds.
* **Interactive Controls:**
    * **Play / Pause** buttons to control the video stream.
    * **Confidence Slider** to filter out low-confidence detections.
* **Performance Optimized:** Only runs the YOLO model **every 5 frames** to ensure a high FPS and smooth user experience. The last known detections are displayed on intermediate frames.
* **Detection Summary:** When paused or finished, the app displays a table summarizing the **total unique count** for each object class (e.g., "Person: 5", "Car: 2").
* **CSV Export:** Instantly download the final object count summary as an `object_counts.csv` file.



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
    This repository includes a `requirements.txt` file with all necessary packages. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `ultralytics` will also install `torch` and `torchvision`, which may take a few minutes.)*

## 🚀 How to Run

1.  From your terminal (with the virtual environment activated), run the Streamlit app. (Assuming you saved the script as `app.py`):
    ```bash
    streamlit run app.py
    ```

2.  Streamlit will open the application in your default web browser (usually at `http://localhost:8501`).

3.  **To use the app:**
    * Select your **Video Source** in the sidebar (Upload or Webcam).
    * Adjust the **Confidence Threshold** if needed.
    * Click the **▶️ Play** button to start the detection.
    * Click the **⏸️ Pause** button to stop the stream. The **Final Object Counts** table and **Download CSV** button will appear below the video.