# 🚆 Railway Track Fall Detection System Using R-CNN for Infrastructure Safety

## 📌 Project Overview

Railway track failures such as track falls, cracks, and obstacles can cause severe accidents if not detected in time.  
This project presents an AI-based railway track fall detection system using **Region-Based Convolutional Neural Networks (R-CNN)** to automatically identify defective track conditions from images.

The system detects faulty regions, highlights defects using **bounding boxes**, and triggers an **immediate alert message with a siren sound** when a defect is identified.  
This approach enhances railway infrastructure safety and reduces reliance on manual inspection.

---

## 🎯 Objectives

- Detect railway track falls, cracks, and obstacles  
- Automatically locate defective regions using bounding boxes  
- Reduce dependency on manual railway inspections  
- Provide instant alert and warning messages  
- Improve railway infrastructure safety through early detection  

---

## 🧠 Technologies Used

- Python  
- TensorFlow & Keras  
- Region-Based Convolutional Neural Network (R-CNN / Faster R-CNN)  
- Flask (Web Framework)  
- OpenCV  
- HTML, CSS  
- LabelImg (for image annotation)  

---

## 📂 Dataset Description

The model is trained using an **image-based dataset** organized into two categories:

- **Defective** – Images showing railway track falls, cracks, or obstacles  
- **Non-Defective** – Images showing normal railway track conditions  

Images are preprocessed using:
- Resizing  
- Normalization  
- Data augmentation  

These steps improve model robustness and detection accuracy under different environmental conditions.

---

## ⚙️ Proposed Methodology

The system follows the steps below:

### 1️⃣ Image Acquisition
Railway track images are captured using cameras or inspection devices.

### 2️⃣ Preprocessing
Images are resized, normalized, and enhanced to improve visibility of defects.

### 3️⃣ Region Proposal Generation
R-CNN generates region proposals that may contain defects.

### 4️⃣ Feature Extraction
CNN layers extract meaningful features such as cracks, gaps, and track misalignment.

### 5️⃣ Classification & Localization
Regions are classified as **Defective** or **Non-Defective**, and bounding boxes are drawn around detected defects.

### 6️⃣ Alert Generation
When a defective track is detected, the system generates an **alert message and siren sound** to warn railway authorities.

---

## 🔔 Alert & Safety Notification System

A major highlight of this project is the **real-time alert mechanism**.

When a railway track defect is detected:
- A siren alarm is triggered  
- A warning message is displayed on the screen  
- The defective region is clearly highlighted  
- Immediate preventive action can be taken  

This alert-based response helps prevent accidents and enhances real-time safety awareness.

---

## 🌐 Web Application

A **Flask-based web application** is developed to provide an interactive interface for railway track monitoring.

The application allows users to:
- Upload railway track images  
- Perform real-time defect detection  
- View detection results with bounding boxes  
- Receive instant alert messages and siren warnings  

---

## 🧪 Testing & Results

The system was tested using multiple railway track images.

### Observations:
- Accurate detection of track falls, cracks, and obstacles  
- Clear visualization of defects using bounding boxes  
- Immediate alert generation for defective tracks  
- Effective performance for infrastructure safety monitoring  

---

## 🖥️ Hardware & Software Requirements

### Hardware
- Camera / Webcam  
- Laptop or PC (minimum 8 GB RAM)  
- Speaker or buzzer for siren alert  
- GPU (optional, for faster training)  

### Software
- Windows / Linux  
- Python 3.x  
- TensorFlow, OpenCV  
- VS Code / Jupyter Notebook  

---

## 🔮 Future Enhancements

- Integration with live CCTV or drone surveillance  
- Automated SMS / email alert notifications  
- Cloud-based deployment  
- Improved accuracy with larger annotated datasets  

---

## 👩‍💻 Author

**Dande Vishnu Priya**  
B.Tech – Artificial Intelligence & Machine Learning  
St. Ann’s College of Engineering and Technology, Chirala  

## 🔗 LinkedIn

https://www.linkedin.com/in/vishnupriyadande
