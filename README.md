#  Automatic Number Plate Recognition (ANPR) System using YOLOv5 and Flask

A complete end-to-end solution for detecting and recognizing vehicle number plates using **YOLOv5** for object detection and **Flask** for web deployment. This project leverages **Computer Vision**, **Deep Learning**, and **Web Technologies** to create a modern and scalable ANPR pipeline.

---

##  Tech Stack & Concepts Used

### Artificial Intelligence & Computer Vision
- **YOLOv5**: Real-time object detection for license plates
- **OpenCV**: Image pre-processing and manipulation
- **OCR (EasyOCR/Tesseract)**: Extracting alphanumeric data from plates
- **Image Augmentation**: For training robustness (if training implemented)

###  Backend & Model Integration
- **Python**: Core programming language
- **PyTorch**: For running YOLOv5 models
- **Flask**: Lightweight server to host ANPR pipeline via a web interface
- **RESTful API Design** (optional): For scalable integration with other systems

###  Frontend (via Flask)
- **HTML/CSS Templates**: For user image upload and result display
- **Static File Handling**: For uploading images and displaying detections

###  Data Handling & Results
- `labels.csv`: To store predictions and evaluations
- Uploaded & processed images stored in `/upload` and `/images`

---

## ️ Project Demo

> Upload an image through the web interface, and the system will:
1. Detect the vehicle number plate using YOLOv5
2. Crop and extract the plate
3. Run OCR to extract the number
4. Display the results back to the user

<img src="static/demo.gif" width="600"/>

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/anpr-yolov5.git
cd anpr-yolov5
