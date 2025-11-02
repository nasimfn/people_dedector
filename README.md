# 👁️ People Detector (OpenCV + HOG + SVM)

This repository contains a simple yet effective **people detection system** using **OpenCV’s HOG (Histogram of Oriented Gradients)** and a **pre-trained SVM** classifier.  
It can detect people in static images or in real-time from your webcam.

---

## 🚀 Features

- Detects people in images or webcam feed  
- Uses OpenCV’s default HOG + SVM people detector  
- Applies Non-Max Suppression (via `imutils`) to remove overlapping boxes  
- Optionally saves detection results as images  
- Lightweight and easy to use  

# Example Usage
### Detect people in an image and show results
python detector.py --image examples/sample.jpg

### Detect people with camera
python detector.py --camera true

### Detect and save result
python detector.py --image examples/sample.jpg --save true
