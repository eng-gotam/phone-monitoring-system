# 📱 Phone Detection & Monitoring System

A Computer Vision system that detects mobile phones in videos, tracks them using ByteTrack, measures how long each phone remains visible, and generates monitoring events.

## 🚀 Features

- Phone detection using YOLOv8 Nano
- Object tracking using ByteTrack
- Phone tracking IDs
- Detection duration monitoring
- Configurable alert threshold
- Warning for long-duration phone detection
- CSV event logging
- Streamlit web interface
- Processed video output

## 🧠 Model Information

| Component | Details |
|---|---|
| Model | YOLOv8 Nano |
| Task | Object Detection |
| Class | Phone |
| Tracker | ByteTrack |
| Framework | Ultralytics YOLO |
| Interface | Streamlit |

## 📊 Dataset

The dataset contains:

- Total Images: 1,674
- Training Images: 1,218
- Validation Images: 270
- Test Images: 186
- Classes: 1
- Target Class: Phone

The original annotations were cleaned and converted into a single-class dataset where:

0 = phone

## 🎯 Model Training

The YOLOv8 Nano model was trained for:

- Epochs: 10
- Image Size: 640
- Batch Size: 8
- Training Device: CPU

### Validation Results

| Metric | Score |
|---|---:|
| Precision | 86.7% |
| Recall | 81.2% |
| mAP@50 | 85.5% |
| mAP@50-95 | 60.4% |

## 🔄 System Architecture

```text
Input Video
     ↓
OpenCV
     ↓
YOLOv8 Nano
     ↓
Phone Detection
     ↓
ByteTrack
     ↓
Tracking ID
     ↓
Duration Monitoring
     ↓
Alert System
     ↓
CSV Event Log
     ↓
Streamlit Dashboard



⏱️ Monitoring Logic

The system tracks each detected phone using a unique tracking ID.

For example:

Phone ID: 1
Start Time: 0.00 sec
End Time: 6.00 sec
Duration: 6.00 sec

If a phone remains detected longer than the configured threshold, the system generates a warning.

Default alert threshold:


3 seconds



📁 Project Structure
phone-monitoring-system/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
│
├── outputs/
└── sample/



🛠️ Technologies
Python
OpenCV
YOLOv8
Ultralytics
ByteTrack
Pandas
Streamlit



▶️ How to Run

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

python -m streamlit run app.py

Then upload a video through the web interface.




📌 Future Improvements
Real-time webcam monitoring
Email/SMS alerts
Database storage
Multiple object classes
Improved tracking
Deployment to cloud